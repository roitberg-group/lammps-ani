# This script runs automatically after molfind if the 'trackmol' task option is chosen
# The point is just to concatenate outputs to a single xyz per frame

from pathlib import Path
from collections import defaultdict
import glob
import numpy as np

def parse_xyz_new_format(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f if line.strip()]

    natoms     = int(lines[0])
    atom_lines = lines[2:2 + natoms]

    parsed = []
    for line in atom_lines:
        line = line.lstrip()
        if '# index:' in line:
            parts     = line.split('# index:')
            atom_line = parts[0].rstrip()
            index     = int(parts[1].strip())
        else:
            atom_line = line
            index     = None
        parsed.append((index, atom_line))
    return parsed

def combine_xyz_per_frame(output_dir: Path, tracked_indices: set):
    # 1) group files by frame
    xyz_files   = glob.glob(str(output_dir / "*_frame_*.xyz"))
    frame_groups = defaultdict(list)
    for fname in xyz_files:
        path = Path(fname)
        try:
            frame_id = path.name.split("_frame_")[1].split(".")[0]
            frame_groups[frame_id].append(path)
        except IndexError:
            continue

    # 2) pre-scan: compute global centroid & per-frame counts
    sum_xyz     = np.zeros(3, dtype=float)
    total_count = 0
    frame_counts = {}
    for frame_id, files in frame_groups.items():
        count = 0
        for fname in files:
            for _, atom_line in parse_xyz_new_format(fname):
                # atom_line: "Element x y z [..]"
                x, y, z = map(float, atom_line.split()[1:4])
                sum_xyz += (x, y, z)
                total_count += 1
                count += 1
        frame_counts[frame_id] = count

    if total_count == 0:
        raise RuntimeError("No atoms found in any XYZ files under " + str(output_dir))

    centroid  = sum_xyz / total_count
    max_atoms = max(frame_counts.values())

    # 3) make output directory
    combined_dir = output_dir / "combined_xyzs"
    combined_dir.mkdir(exist_ok=True)
    traj_path = combined_dir / "all_frames_combined.xyz"
    traj_file_handle = open(traj_path, "w")

    # 4) second pass: combine, sort, pad, write
    for frame_id, files in sorted(frame_groups.items(), key=lambda x: int(x[0])):
        flat_data = []
        for fname in files:
            flat_data.extend(parse_xyz_new_format(fname))

        # sort tracked indices first, then by index ascending
        flat_data_sorted = sorted(
            flat_data,
            key=lambda x: (x[0] is None, x[0] if x[0] is not None else 0)
        )

        n_real    = len(flat_data_sorted)
        n_missing = max_atoms - n_real

        output_file = combined_dir / f"frame_{frame_id}_combined.xyz"
        with open(output_file, "w") as out:
            # header: real + dummy
            out.write(f"{n_real + n_missing}\n")
            out.write(
                "Fragments: "
                + " ".join(p.name.split('_target')[0] for p in files)
                + "\n"
            )

            # write the real atoms (with any index-comments)
            for idx, atom_line in flat_data_sorted:
                comment = f"    # Index {idx}" if idx in tracked_indices else ""
                out.write(f"{atom_line}{comment}\n")

            # pad with dummy sulfurs at the global centroid
            cx, cy, cz = centroid
            for _ in range(n_missing):
                out.write(
                    f"S {cx:.6f} {cy:.6f} {cz:.6f}    # dummy sulfur anchor\n"
                )

        print(f"[cat_xyz] Wrote {output_file}")

        with open(output_file) as cf:
            traj_file_handle.write(cf.read())
    
    traj_file_handle.close()
    print(f"[cat_xyz] Wrote multi-frame trajectory: {traj_path}")

