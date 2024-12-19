import argparse
from pathlib import Path
import mdtraj as md
from decimal import Decimal, getcontext

getcontext().prec = 12  # This sets the precision to 12 decimal places


def parse_arguments():
    parser = argparse.ArgumentParser(description='Split a large trajectory file into smaller segments.')
    parser.add_argument('traj_file', type=str, help='Path to the trajectory file')
    parser.add_argument('top_file', type=str, help='Path to the topology file')
    parser.add_argument('--segment_duration_ns', type=float, default=0.1, help='Segment duration in nanoseconds')
    parser.add_argument('--dump_interval', type=int, default=50, help='Dump interval')
    parser.add_argument('--timestep', type=float, default=0.25, help='Time step in femtoseconds')
    parser.add_argument('--output_dir', type=Path, default=None, help='Output directory')
    return parser.parse_args()


def calculate_frames_per_segment(timestep, segment_duration_ns, dump_interval):
    frames = (segment_duration_ns * 1e6) / (timestep * dump_interval)
    if not frames.is_integer() or frames % 100 != 0:
        raise ValueError(f"The calculated number of frames per segment must be an integer and a multiple of 100, found: {frames}")
    return int(frames)


def split_trajectory(traj_file, top_file, timestep, output_dir, frames_per_segment, dump_interval):
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, traj in enumerate(md.iterload(traj_file, top=top_file, chunk=frames_per_segment)):
        time_offset = float(i * Decimal(frames_per_segment) * Decimal(dump_interval) * Decimal(timestep) * Decimal('1e-6'))
        print(f"Saving segment {i} with time offset {time_offset} ns")
        segment_filename = f"{Path(traj_file).stem}_{time_offset}ns.dcd"
        traj.save_dcd(output_dir / segment_filename)
        if time_offset > 1.0:
            break


def main():
    args = parse_arguments()
    frames_per_segment = calculate_frames_per_segment(args.timestep, args.segment_duration_ns, args.dump_interval)
    print(f"Splitting every {frames_per_segment} frames ({args.segment_duration_ns} ns)")

    output_dir = Path(args.output_dir) if args.output_dir else Path(f"{args.traj_file}_split")
    split_trajectory(args.traj_file, args.top_file, args.timestep, output_dir, frames_per_segment, args.dump_interval)


if __name__ == "__main__":
    main()
