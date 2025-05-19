# This script runs automatically after molfind if the 'trackmol' task option is chosen
# The point is just to concatenate outputs to a single xyz per frame


from pathlib import Path
from collections import defaultdict
import glob

def parse_xyz_new_format(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f if line.strip()]

    natoms = int(lines[0])
    atom_lines = lines[2:2 + natoms]

    parsed = []
    for line in atom_lines:
        line = line.lstrip()
        if '# index:' in line:
            parts = line.split('# index:')
            atom_line = parts[0].rstrip()
            index = int(parts[1].strip())
        else:
            atom_line = line
            index = None
        parsed.append((index, atom_line))
    return parsed

def combine_xyz_per_frame(output_dir: Path, tracked_indices: set):
    xyz_files = glob.glob(str(output_dir / "*_frame_*.xyz"))
    frame_groups = defaultdict(list)

    for fname in xyz_files:
        path = Path(fname)
        try:
            frame_id = path.name.split("_frame_")[1].split(".")[0]
            frame_groups[frame_id].append(path)
        except IndexError:
            continue

    combined_dir = output_dir / "combined_xyzs"
    combined_dir.mkdir(exist_ok=True)

    for frame_id, files in frame_groups.items():
        flat_data = []
        for fname in files:
            fragment_data = parse_xyz_new_format(fname)
            flat_data.extend(fragment_data)

        output_file = combined_dir / f"frame_{frame_id}_combined.xyz"
        with open(output_file, "w") as out:
            out.write(f"{len(flat_data)}\n")
            out.write(f"Fragments: {' '.join([f.name.split('_target')[0] for f in files])}\n")
            for idx, atom_line in flat_data:
                comment = f"    # Index {idx}" if idx in tracked_indices else ""
                out.write(f"{atom_line}{comment}\n")

        print(f"[cat_xyz] Wrote {output_file}")

