import glob
import os
import re


def extract_data_from_log(file_path):
    """
    Extract metrics from a single log file
    """
    try:
        with open(file_path, "r") as f:
            content = f.read()

        data = {
            "file_name": os.path.basename(file_path),
            "epoch_time": None,
            "peak_memory": None,
            "mx_queries": None,
            "trainable_params": None,
            "dataset": None,
            "n_gaussians": None,
            "ct": None,
            "ft": None,
        }

        # Extract from command line (first line)
        first_line_match = re.search(
            r"--dataset\s+(\w+).*?--n-gaussians\s+(\d+).*?-ct\s+(\d+).*?-ft\s+(\d+)",
            content,
        )
        if first_line_match:
            data["dataset"] = first_line_match.group(1)
            data["n_gaussians"] = int(first_line_match.group(2))
            data["ct"] = int(first_line_match.group(3))
            data["ft"] = int(first_line_match.group(4))

        # Extract trainable parameters
        params_match = re.search(r"Trainable params:\s*([\d,]+)", content)
        if params_match:
            data["trainable_params"] = int(params_match.group(1).replace(",", ""))

        # Extract epoch time
        epoch_time_match = re.search(r"epoch_time:\s*([\d.]+)", content)
        if epoch_time_match:
            data["epoch_time"] = float(epoch_time_match.group(1))

        # Extract peak memory
        peak_mem_match = re.search(r"Peak memory:\s*([\d.]+)\s*MB", content)
        if peak_mem_match:
            data["peak_memory"] = float(peak_mem_match.group(1))

        # Extract mx queries
        mx_queries_match = re.search(r"mx queries:\s*([\d.]+)", content)
        if mx_queries_match:
            data["mx_queries"] = float(mx_queries_match.group(1))

        return data

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def extract_all_logs(log_directory=None, log_files=None):
    """
    Extract data from all log files
    """
    if log_files is None:
        # Get all .log files in directory
        log_files = glob.glob(os.path.join("results/var_d/", "*.log"))

    all_data = []

    for log_file in log_files:
        print(f"Processing: {log_file}")
        data = extract_data_from_log(log_file)
        if data:
            all_data.append(data)

    return all_data


def display_extracted_data(all_data):
    """
    Display the extracted data in a readable format
    """
    print("\n" + "=" * 120)
    print(
        f"{'File':<30} {'Dataset':<12} {'CT':<4} {'FT':<4} {'Gaussians':<10} {'Params':<12} {'Epoch Time':<12} {'Peak Mem (MB)':<15} {'mx queries':<12}"
    )
    print("=" * 120)

    for data in all_data:
        print(
            f"{data['file_name']:<30} "
            f"{data['dataset']:<12} "
            f"{data['ct']:<4} "
            f"{data['ft']:<4} "
            f"{data['n_gaussians']:<10} "
            f"{data['trainable_params']:<12} "
            f"{data['epoch_time']:<12.2f} "
            f"{data['peak_memory']:<15.2f} "
            f"{data['mx_queries']:<12.1f}"
        )


def save_to_csv(all_data, output_file="extracted_metrics.csv"):
    """
    Save extracted data to CSV file
    """
    import csv

    with open(output_file, "w", newline="") as csvfile:
        fieldnames = [
            "file_name",
            "dataset",
            "ct",
            "ft",
            "n_gaussians",
            "trainable_params",
            "epoch_time",
            "peak_memory",
            "mx_queries",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for data in all_data:
            writer.writerow(data)

    print(f"\nData saved to {output_file}")


# Main execution
if __name__ == "__main__":
    # Extract data from all log files
    all_data = extract_all_logs()

    # Display results
    # display_extracted_data(all_data)

    # Save to CSV
    save_to_csv(all_data, "var_d_ablation.csv")

    # Print summary
    print(f"\nSummary: Extracted data from {len(all_data)} log files")

    # Check for missing data
    missing_fields = []
    for field in ["epoch_time", "peak_memory", "mx_queries", "trainable_params"]:
        missing = sum(1 for data in all_data if data[field] is None)
        if missing > 0:
            missing_fields.append((field, missing))

    if missing_fields:
        print("\nMissing data:")
        for field, count in missing_fields:
            print(f"  {field}: {count} files")
