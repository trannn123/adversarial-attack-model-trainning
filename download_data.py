import os
import thrember


def main():
    print("=== DOWNLOAD EMBER2024 – WIN32 ONLY ===")

    # Thư mục lưu dữ liệu
    data_dir = "./data"

    # Tạo thư mục nếu chưa có
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"[+] Created directory: {data_dir}")
    else:
        print(f"[+] Using existing directory: {data_dir}")

    # Cấu hình download
    split = "all"        # train + test (+ challenge nếu có)
    file_type = "Win32"  # chỉ Win32 (nhẹ nhất trong PE)

    print(f"[+] Download config:")
    print(f"    Split    : {split}")
    print(f"    File type: {file_type}")
    print("[+] Starting download...")

    # Gọi hàm download
    thrember.download_dataset(
        data_dir,
        split=split,
        file_type=file_type
    )

    print("=== DOWNLOAD COMPLETED ===")


if __name__ == "__main__":
    main()
