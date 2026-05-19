"""
test_code.py
Incremental test script for the autoencoder compression pipeline.
Processes first N frames (default 100) with atomic, testable steps.
Focus: Diagnose range coding issues with proper per-channel CDF tables.
Includes optional socket-based network transfer test.

Usage:
    python scripts/test_code.py --input_folder data/processed_frames/<FOLDER> --run ml/models/saved/<RUN> --num_frames 100
    python scripts/test_code.py --input_folder data/processed_frames/<FOLDER> --run ml/models/saved/<RUN> --num_frames 100 --test_network
"""

import os
import time
import glob
import argparse
import socket
import struct
import threading
import numpy as np
import cv2
import torch
from tqdm import tqdm

from ml.models.entropy import FactorizedBottleneck
from ml.utils.coding import (
    RangeEncoder, RangeDecoder,
    build_cdf_tables_from_bottleneck,
    quantize_latents_to_symbols
)
from ml.utils.constants import PROJECT_ROOT, get_device
from ml.utils.model_loading import detect_latent_channels, load_model

DEVICE = get_device()


# -----------------------------------------------------------------------------
# Atomic Helper Functions
# -----------------------------------------------------------------------------

def setup_paths(args):
    """Resolve model and frame paths, validate files. Returns dict."""
    result = {}

    # Resolve run path
    if not args.run:
        runs = sorted([f.path for f in os.scandir(os.path.join(PROJECT_ROOT, 'ml', 'models', 'saved')) if f.is_dir()])
        if not runs:
            raise FileNotFoundError("No runs found in ml/models/saved/")
        result['run_path'] = runs[-1]
    else:
        result['run_path'] = args.run

    model_path = os.path.join(result['run_path'], "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    result['model_path'] = model_path

    # Resolve input folder
    if not args.input_folder:
        folder_candidates = sorted([f.path for f in os.scandir(os.path.join(PROJECT_ROOT, 'data', 'processed_frames')) if f.is_dir()])
        if not folder_candidates:
            raise FileNotFoundError("No input folder found in data/processed_frames/")
        result['input_folder'] = folder_candidates[0]
    else:
        result['input_folder'] = args.input_folder

    # Get frame files (first N)
    frame_files = sorted(glob.glob(os.path.join(result['input_folder'], "*.jpg")))
    if not frame_files:
        raise FileNotFoundError(f"No .jpg files found in {result['input_folder']}")
    result['frame_files'] = frame_files[:args.num_frames]

    return result


def preprocess_frame(frame_path, prev_frame_tensor, frame_count, iframe_interval, device):
    """
    Load JPG, handle I/P-frame logic, normalize to tensor.
    Returns dict with frame data and metadata.
    """
    frame = cv2.imread(frame_path)
    if frame is None:
        return None

    h, w = frame.shape[:2]
    original_bytes = frame.nbytes

    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    frame_tensor = frame_tensor.to(device)

    is_iframe = (frame_count % iframe_interval == 0)
    if is_iframe:
        input_to_model = frame_tensor
        frame_type = "I-frame"
    else:
        if prev_frame_tensor is None:
            input_to_model = frame_tensor
            frame_type = "I-frame"
        else:
            input_to_model = frame_tensor - prev_frame_tensor
            frame_type = "P-frame"

    return {
        'frame': frame,
        'frame_tensor': frame_tensor,
        'input_to_model': input_to_model,
        'frame_type': frame_type,
        'h': h,
        'w': w,
        'original_bytes': original_bytes,
        'next_prev_frame': frame_tensor.clone()
    }


def encode_latents(model, input_tensor):
    """
    Run encoder + bottleneck to get float latents and likelihood.
    Returns dict with y_q_float, p_y, y_q_shape, bottleneck_obj.
    """
    with torch.no_grad():
        y = model.encoder(input_tensor)
        y_q_float, p_y = model.bottleneck(y, training=False)

    return {
        'y_q_float': y_q_float,
        'p_y': p_y,
        'y_q_shape': y_q_float.shape,
        'bottleneck': model.bottleneck
    }


def build_cdf_tables(bottleneck, symbol_range=256):
    """
    Build per-channel CDF tables from FactorizedBottleneck.
    Returns (cdf_tables, symbol_offset).
    """
    cdf_tables, symbol_offset = build_cdf_tables_from_bottleneck(bottleneck, symbol_range)
    print(f"[CDF] Built tables shape: {cdf_tables.shape}, symbol_offset: {symbol_offset}")
    return cdf_tables, symbol_offset


def quantize_to_symbols(y_q_float, symbol_offset, symbol_range):
    """
    Convert float latents to integer symbols in [0, symbol_range).
    Returns (symbols_int, y_q_shape).
    """
    symbols = quantize_latents_to_symbols(y_q_float, symbol_offset, symbol_range)
    y_q_shape = y_q_float.shape
    return symbols, y_q_shape


def calculate_sizes(y_q_float, compressed_bytes, cdf_tables, symbol_offset):
    """
    Measure payload bloat sources.
    Returns dict with size metrics.
    """
    y_q_np = y_q_float.cpu().numpy()
    raw_latent_bytes = y_q_np.nbytes
    compressed_size = len(compressed_bytes)
    cdf_size = cdf_tables.nbytes

    # Old approach: flatten p_y (which is wrong anyway) and send as float16
    # Simulate old approach size
    fake_old_cdf = np.random.rand(*y_q_np.shape).astype(np.float16)
    old_cdf_size = fake_old_cdf.nbytes
    old_total_payload = compressed_size + old_cdf_size

    return {
        'raw_latent_bytes': raw_latent_bytes,
        'compressed_size': compressed_size,
        'cdf_tables_size': cdf_size,
        'new_total_payload': compressed_size + cdf_size,
        'old_cdf_size': old_cdf_size,
        'old_total_payload': old_total_payload,
        'compression_ratio_raw': raw_latent_bytes / compressed_size if compressed_size > 0 else 0,
        'compression_ratio_total': raw_latent_bytes / (compressed_size + cdf_size) if (compressed_size + cdf_size) > 0 else 0
    }


def reconstruct_frame(model, decoded_latents, y_q_shape, device):
    """
    Run decoder on recovered latents, postprocess to frame.
    Returns reconstructed_frame (numpy).
    """
    y_q = torch.from_numpy(decoded_latents).reshape(y_q_shape).float().to(device)

    with torch.no_grad():
        reconstructed = model.decoder(y_q)

    out_frame = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_frame = (out_frame * 255).clip(0, 255).astype(np.uint8)
    return out_frame


# -----------------------------------------------------------------------------
# Network Transfer Functions
# -----------------------------------------------------------------------------

def receiver_thread(receiver_port, output_video, target_fps, done_event, start_event, 
                   model_path, latent_channels, device, num_frames):
    """
    Thread that listens for connections, receives, decodes, and saves frames.
    Loads model locally (no CDF transmission needed).
    """
    # Load model using the factory function
    model = load_model(model_path, latent_channels, device)
    
    # Build CDF tables locally
    cdf_tables, symbol_offset = build_cdf_tables_from_bottleneck(model.bottleneck)
    print(f"[Receiver] Model loaded on {device}, CDF tables: {cdf_tables.shape}")
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('localhost', receiver_port))
    server_socket.listen(1)
    print(f"[Receiver] Listening on port {receiver_port}")
    
    start_event.set()  # Signal that receiver is ready
    
    conn, addr = server_socket.accept()
    print(f"[Receiver] Connected from {addr}")
    
    h, w = None, None
    fourcc = None
    out_writer = None
    frame_idx = 0
    cumulative_bytes = 0
    frames_received = 0
    
    try:
        while frames_received < num_frames:
            # Read header: frame_idx (4 bytes) + data_size (4 bytes)
            header = b''
            while len(header) < 8:
                data = conn.recv(8 - len(header))
                if not data:
                    break
                header += data
            
            if len(header) < 8:
                break
                
            recv_frame_idx, data_size = struct.unpack('ii', header)
            
            if recv_frame_idx == -1:  # End marker
                print(f"[Receiver] Received end marker")
                break
            
            # Read payload
            payload = b''
            while len(payload) < data_size:
                chunk = conn.recv(data_size - len(payload))
                if not chunk:
                    break
                payload += chunk
            
            # Unpack header: frame_idx(1) + y_q_shape(4) + frame_shape(2) + compressed_size(1)
            # Format 'iiiiiiii' = 8 integers = 32 bytes
            header_data = struct.unpack('iiiiiiii', payload[:32])
            y_q_shape = (header_data[1], header_data[2], header_data[3], header_data[4])
            orig_h, orig_w = header_data[5], header_data[6]
            compressed_size = header_data[7]
            
            # Extract compressed bytes
            compressed_bytes = payload[32:32+compressed_size]
            
            print(f"[Receiver] Frame {recv_frame_idx}: compressed_size={compressed_size}, payload_len={len(payload)}")
            
            # Decode using range coding
            decoder = RangeDecoder()
            decoded_symbols = decoder.decode(compressed_bytes, cdf_tables, y_q_shape)
            
            # Convert back to latents (subtract offset, reshape)
            decoded_latents = decoded_symbols.reshape(y_q_shape) - symbol_offset
            
            if h is None:
                h, w = orig_h, orig_w
                print(f"[Receiver] Output resolution: {w}x{h}")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(output_video, fourcc, target_fps, (w, h))
            
            # Reconstruct frame
            reconstructed = reconstruct_frame(model, decoded_latents, y_q_shape, device)
            
            out_writer.write(reconstructed)
            
            # Log reconstructed bytes
            reconstructed_bytes = reconstructed.nbytes
            cumulative_bytes += reconstructed_bytes
            
            frame_idx += 1
            frames_received += 1
        
    except Exception as e:
        print(f"[Receiver] Error: {e}")
    finally:
        if out_writer:
            out_writer.release()
        conn.close()
        server_socket.close()
        print(f"[Receiver] Video saved to {output_video}")
        done_event.set()


# -----------------------------------------------------------------------------
# Main Workflow
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test autoencoder compression pipeline (first N frames)")
    parser.add_argument("--input_folder", type=str, default=None, help="Path to folder with JPG frames")
    parser.add_argument("--run", type=str, default=None, help="Path to run folder (contains best_model.pth)")
    parser.add_argument("--output", type=str, default="data/reconstructed_test.mp4", help="Path to output video file")
    parser.add_argument("--target_fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--iframe_interval", type=int, default=10, help="Every Nth frame is an I-frame")
    parser.add_argument("--num_frames", type=int, default=100, help="Number of frames to process (default 100)")
    parser.add_argument("--symbol_range", type=int, default=256, help="Number of possible symbol values for range coding")
    parser.add_argument("--latent_channels", type=int, default=128, help="Number of latent channels in model")
    parser.add_argument("--test_network", action="store_true", help="Test with socket-based network transfer")
    parser.add_argument("--sender_port", type=int, default=9999, help="Port for sender to connect to")
    parser.add_argument("--receiver_port", type=int, default=9998, help="Port for receiver to listen on")
    args = parser.parse_args()

    print("="*60)
    print("AUTOENCODER COMPRESSION PIPELINE TEST")
    if args.test_network:
        print("MODE: Network Transfer Test (localhost sockets)")
    print("="*60)

    # Step1: Setup paths
    print("\n[Step 1] Setting up paths...")
    paths = setup_paths(args)
    print(f"  Run path: {paths['run_path']}")
    print(f"  Input folder: {paths['input_folder']}")
    print(f"  Frame count: {len(paths['frame_files'])}")

    # Step 2: Load model
    print("\n[Step 2] Detecting and loading model...")
    detected_channels = detect_latent_channels(paths['model_path'])
    print(f"  Detected latent_channels: {detected_channels}")
    model = load_model(paths['model_path'], detected_channels, DEVICE)

    # Step 3: Build CDF tables (once for all frames, since bottleneck is fixed)
    print("\n[Step 3] Building CDF tables from bottleneck...")
    cdf_tables, symbol_offset = build_cdf_tables(model.bottleneck, args.symbol_range)
    print(f"  CDF tables shape: {cdf_tables.shape}")
    print(f"  Symbol offset: {symbol_offset}")
    print(f"  CDF tables size: {cdf_tables.nbytes} bytes")

    if args.test_network:
        # Network test mode: start receiver thread, then send frames
        done_event = threading.Event()
        start_event = threading.Event()
        
        receiver_thread_handle = threading.Thread(
            target=receiver_thread,
            args=(args.receiver_port, args.output, args.target_fps, done_event, start_event,
                  paths['model_path'], detected_channels, DEVICE, len(paths['frame_files']))
        )
        receiver_thread_handle.start()
        
        # Wait for receiver to be ready
        start_event.wait()
        time.sleep(0.5)  # Small delay to ensure socket is ready
        
        # Connect as sender
        print(f"\n[Sender] Connecting to receiver on port {args.receiver_port}...")
        sender_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sender_socket.connect(('localhost', args.receiver_port))
        except ConnectionRefusedError:
            print(f"[Sender] Could not connect to receiver on port {args.receiver_port}")
            return
        
        print(f"[Sender] Connected. Starting transfer...")
        
        # Process and send frames
        prev_frame_tensor = None
        frame_count = 0
        stats = []
        pbar = tqdm(total=len(paths['frame_files']), desc="Sending Frames")
        
        for frame_path in paths['frame_files']:
            frame_data = preprocess_frame(
                frame_path, prev_frame_tensor, frame_count,
                args.iframe_interval, DEVICE
            )
            if frame_data is None:
                frame_count += 1
                pbar.update(1)
                continue
            
            # Encode latents
            latent_data = encode_latents(model, frame_data['input_to_model'])
            
            # Quantize to symbols
            symbols_int, y_q_shape = quantize_to_symbols(
                latent_data['y_q_float'], symbol_offset, args.symbol_range
            )
            
            # Encode symbols using range coding
            encoder = RangeEncoder()
            compressed_bytes = encoder.encode(symbols_int, cdf_tables, y_q_shape)
            
            # Pack payload: header (32 bytes) = frame_idx(1) + y_q_shape(4) + frame_shape(2) + compressed_size(1)
            header = struct.pack('iiiiiiii',
                frame_count,
                y_q_shape[0], y_q_shape[1], y_q_shape[2], y_q_shape[3],
                frame_data['h'], frame_data['w'],
                len(compressed_bytes)
            )
            payload = header + compressed_bytes
            
            # Send over network: frame_idx + data_size + payload
            send_header = struct.pack('ii', frame_count, len(payload))
            sender_socket.sendall(send_header + payload)
            
            # Record stats
            stats.append({
                'frame': frame_count,
                'type': frame_data['frame_type'],
                'original_bytes': frame_data['original_bytes'],
                'compressed_size': len(compressed_bytes),
                'total_sent': len(payload)
            })
            
            prev_frame_tensor = frame_data['next_prev_frame']
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        
        # Send end marker
        try:
            end_marker = struct.pack('ii', -1, 0)
            sender_socket.sendall(end_marker)
        except:
            pass
        
        sender_socket.close()
        
        # Wait for receiver to finish
        done_event.wait(timeout=30)
        
        # Print summary
        print("\n" + "="*60)
        print("NETWORK TRANSFER COMPLETE")
        print("="*60)
        total_original = sum(s['original_bytes'] for s in stats)
        total_compressed = sum(s['compressed_size'] for s in stats)
        total_sent = sum(s['total_sent'] for s in stats)
        print(f"\nTotal Original Bytes:      {total_original:,}")
        print(f"Total Compressed:         {total_compressed:,}")
        print(f"Total Sent (with header): {total_sent:,}")
        print(f"Compression Ratio: {total_original / total_sent:.2f}x" if total_sent > 0 else "")
        
    else:
        # Local test mode (original behavior)
        print(f"\n[Step 4] Processing {len(paths['frame_files'])} frames...")
        prev_frame_tensor = None
        frame_count = 0
        stats = []
        pbar = tqdm(total=len(paths['frame_files']), desc="Testing Frames")

        for frame_path in paths['frame_files']:
            # 4a: Preprocess frame
            frame_data = preprocess_frame(
                frame_path, prev_frame_tensor, frame_count,
                args.iframe_interval, DEVICE
            )
            if frame_data is None:
                frame_count += 1
                pbar.update(1)
                continue

            # --- MEASURE COMPRESSION ---
            comp_start = time.perf_counter()
            # 4b: Encode latents
            latent_data = encode_latents(model, frame_data['input_to_model'])

            # 4c: Quantize to symbols
            symbols_int, y_q_shape = quantize_to_symbols(
                latent_data['y_q_float'], symbol_offset, args.symbol_range
            )
            
            # 4d: Encode symbols (part of range coding)
            encoder = RangeEncoder()
            compressed_bytes = encoder.encode(symbols_int, cdf_tables, y_q_shape)
            comp_end = time.perf_counter()
            comp_ms = (comp_end - comp_start) * 1000

            # --- MEASURE DECOMPRESSION ---
            decomp_start = time.perf_counter()
            # 4e: Decode symbols
            decoder = RangeDecoder()
            decoded_symbols = decoder.decode(compressed_bytes, cdf_tables, y_q_shape)
            
            # 4f: Reconstruct frame
            decoded_latents = decoded_symbols.reshape(y_q_shape) - symbol_offset
            reconstructed = reconstruct_frame(
                model, decoded_latents, y_q_shape, DEVICE
            )
            decomp_end = time.perf_counter()
            decomp_ms = (decomp_end - decomp_start) * 1000

            # Verify integrity
            decode_matches = np.array_equal(symbols_int, decoded_symbols.flatten())

            # 4g: Calculate sizes (for stats)
            size_data = calculate_sizes(
                latent_data['y_q_float'],
                compressed_bytes,
                cdf_tables, symbol_offset
            )
            
            # Debug: Isolate encode/decode for first frame
            if frame_count == 0:
                print(f"\n  DEBUG Frame0: Isolating encode/decode...")
                
                # Check what's actually in symbols_int
                print(f"    symbols_int type: {type(symbols_int)}, shape: {symbols_int.shape}")
                print(f"    symbols_int (first 10): {symbols_int[:10]}")
                print(f"    symbols_int min: {symbols_int.min()}, max: {symbols_int.max()}")
                
                # Check y_q_float
                yqf = latent_data['y_q_float']
                print(f"\n    y_q_float shape: {yqf.shape}")
                print(f"    y_q_float min: {yqf.min().item():.2f}, max: {yqf.max().item():.2f}")
                print(f"    y_q_float (first 5 flat): {yqf.flatten()[:5].cpu().numpy()}")
                
                # What quantize_latents_to_symbols should produce
                y_rounded = torch.round(yqf)
                expected_symbols = (y_rounded + symbol_offset).clamp(0, 255).to(torch.int32).cpu().numpy().flatten()
                print(f"\n    Expected symbols (first 10): {expected_symbols[:10]}")
                print(f"    Expected min: {expected_symbols.min()}, max: {expected_symbols.max()}")
                
                # Direct encode/decode test with symbols_int
                test_symbols = symbols_int.copy()
                test_shape = y_q_shape
                
                enc = RangeEncoder()
                test_compressed = enc.encode(test_symbols, cdf_tables, test_shape)
                print(f"\n    Compressed size: {len(test_compressed)} bytes")
                
                dec = RangeDecoder()
                decoded = dec.decode(test_compressed, cdf_tables, test_shape)
                
                # Check shapes
                print(f"    test_symbols shape: {test_symbols.shape}")
                print(f"    decoded shape: {decoded.shape}")
                print(f"    test_shape: {test_shape}")
                
                matches = np.array_equal(test_symbols, decoded)
                print(f"    Direct encode/decode match: {matches}")
                print(f"    Decoded (first 10): {decoded.flatten()[:10]}")
                print(f"    Decoded min: {decoded.min()}, max: {decoded.max()}")

            # Record stats
            stats.append({
                'frame': frame_count,
                'type': frame_data['frame_type'],
                'original_bytes': frame_data['original_bytes'],
                'raw_latent_bytes': size_data['raw_latent_bytes'],
                'compressed_size': size_data['compressed_size'],
                'cdf_tables_size': size_data['cdf_tables_size'],
                'new_total_payload': size_data['new_total_payload'],
                'old_total_payload': size_data['old_total_payload'],
                'compression_ratio_raw': size_data['compression_ratio_raw'],
                'compression_ratio_total': size_data['compression_ratio_total'],
                'decode_matches': decode_matches,
                'comp_ms': comp_ms,
                'decomp_ms': decomp_ms
            })

            # Print per-frame summary
            if frame_count % 20 == 0:
                print(f"\n  Frame {frame_count} ({frame_data['frame_type']}):")
                print(f"    Original: {frame_data['original_bytes']:,} bytes")
                print(f"    Compressed: {size_data['compressed_size']:,} bytes")
                print(f"    Time: Comp {comp_ms:.2f}ms, Decomp {decomp_ms:.2f}ms")
                print(f"    Decode matches: {decode_matches}")

            prev_frame_tensor = frame_data['next_prev_frame']
            frame_count += 1
            pbar.update(1)

        pbar.close()

        # Step 5: Print summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)

        total_original = sum(s['original_bytes'] for s in stats)
        total_compressed = sum(s['compressed_size'] for s in stats)
        total_payload_new = sum(s['new_total_payload'] for s in stats)
        avg_comp_ms = sum(s['comp_ms'] for s in stats) / len(stats) if stats else 0
        avg_decomp_ms = sum(s['decomp_ms'] for s in stats) / len(stats) if stats else 0
        decode_success_rate = sum(1 for s in stats if s['decode_matches']) / len(stats) * 100 if stats else 0

        print(f"\nProcessed {len(stats)} frames")
        print(f"Model: {paths['model_path']}")
        print(f"\nTotal Original Bytes:      {total_original:,}")
        print(f"Total Compressed (no CDF): {total_compressed:,}")
        print(f"Total New Payload:          {total_payload_new:,} (CDF sent once: {cdf_tables.nbytes:,})")

        print(f"\nPerformance Metrics (Average per Frame):")
        print(f"  Compression:   {avg_comp_ms:.2f} ms")
        print(f"  Decompression: {avg_decomp_ms:.2f} ms")
        print(f"  Total Latency: {avg_comp_ms + avg_decomp_ms:.2f} ms")

        print(f"\nNew Compression Ratio: {total_original / total_payload_new:.2f}x" if total_payload_new > 0 else "")
        print(f"Decode Success Rate:     {decode_success_rate:.1f}%")

        # Save stats
        import pandas as pd
        df = pd.DataFrame(stats)
        output_path = os.path.join(paths['run_path'], "test_code_metrics.csv")
        df.to_csv(output_path, index=False)
        print(f"\nMetrics saved to {output_path}")


if __name__ == "__main__":
    main()
