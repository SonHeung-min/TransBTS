# TransBTS & TransBTSV2: Multimodal Brain Tumor Segmentation

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.7-blue)
![PyTorch](https://img.shields.io/badge/pytorch-1.6.0-green)

> **Lưu ý:** Đây là repository cá nhân được sử dụng để nghiên cứu, triển khai và tùy chỉnh lại mã nguồn gốc của các bài báo khoa học.

Dự án này là cài đặt thực nghiệm cho:
1.  [**TransBTS**](https://arxiv.org/abs/2103.04430): Multimodal Brain Tumor Segmentation Using Transformer (MICCAI 2021).
2.  [**TransBTSV2**](https://arxiv.org/abs/2201.12785): Towards Better and More Efficient Volumetric Segmentation of Medical Images.

---

## 📑 Mục lục (Table of Contents)
- [Cấu trúc dự án](#-cấu-trúc-dự-án-project-structure)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống-requirements)
- [Dữ liệu](#-dữ-liệu-dataset)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng-usage)
    - [1. Tiền xử lý dữ liệu](#1-tiền-xử-lý-dữ-liệu-preprocessing)
    - [2. Huấn luyện mô hình](#2-huấn-luyện-mô-hình-training)
    - [3. Kiểm thử & Đánh giá](#3-kiểm-thử--đánh-giá-testing)
- [Trích dẫn & Bản quyền](#-trích-dẫn--bản-quyền-citation--license)

---

## 📂 Cấu trúc dự án (Project Structure)
```
TransBTS/
├── data/                       # Quản lý dữ liệu & Tiền xử lý
│   ├── BraTS.py                # Dataset loader cho BraTS
│   ├── preprocess.py           # Script tiền xử lý (convert .nii -> .pkl)
│   ├── train.txt               # Danh sách file huấn luyện
│   └── valid.txt               # Danh sách file validation
├── models/                     # Kiến trúc mô hình
│   ├── TransBTS/               # Mã nguồn TransBTS (MICCAI 2021)
│   │   ├── IntmdSequential.py                  # Các lớp trung gian
│   │   ├── PositionalEncoding.py               # Mã hóa vị trí
│   │   ├── TransBTS_downsample8x_skipconnection.py # Kiến trúc chính
│   │   ├── Transformer.py                      # Module Transformer
│   │   ├── Unet_skipconnection.py              # Phần U-Net
│   │   └── README.md
│   ├── TransBTSV2/             # Mã nguồn TransBTSV2
│   │   └── README.md
│   ├── criterions.py           # Các hàm Loss function
│   └── README.md
├── utils/                      # Các hàm tiện ích hỗ trợ
├── figures/                    # Biểu đồ và hình ảnh minh họa
├── train.py                    # Script huấn luyện chính
├── test.py                     # Script kiểm thử/đánh giá
├── predict.py                  # Script dự đoán (inference)
├── LICENSE                     # Thông tin bản quyền
└── README.md                   # Tài liệu hướng dẫn (File này)
```

---

## 🛠 Yêu cầu hệ thống (Requirements)
Để chạy mã nguồn này, vui lòng đảm bảo môi trường đã cài đặt các thư viện sau:
*   Python 3.7
*   PyTorch 1.6.0
*   TorchiVision 0.7.0
*   Pickle
*   Nibabel

Cài đặt nhanh các thư viện phụ thuộc:
```bash
pip install torch==1.6.0 torchvision==0.7.0 nibabel pickle-mixin
```

---

## 💾 Dữ liệu (Dataset)
Các bộ dữ liệu y tế được hỗ trợ và sử dụng trong nghiên cứu này:

| Dataset | Mô tả | Link Tải |
| :--- | :--- | :--- |
| **BraTS 2019/2020** | Khối u não đa phương thức | [Download](https://ipp.cbica.upenn.edu/) |
| **LiTS 2017** | Khối u gan | [Download](https://competitions.codalab.org/competitions/17094) |
| **KiTS 2019** | Khối u thận | [Download](https://kits19.grand-challenge.org/data/) |

---

## 🚀 Hướng dẫn sử dụng (Usage)

### 1. Tiền xử lý dữ liệu (Preprocessing)
Đối với dữ liệu **BraTS** (2019/2020), sau khi tải về, hãy chạy script sau để chuyển đổi file `.nii` sang định dạng `.pkl` tối ưu hóa cho việc load dữ liệu và chuẩn hóa intensity.
**Lưu ý:** Script nằm trong thư mục `data/`. Bạn cần thay đổi đường dẫn (path) trong file `data/preprocess.py` trỏ tới thư mục chứa dữ liệu đã tải về của mình trước khi chạy.

```bash
python3 data/preprocess.py
```

### 2. Huấn luyện mô hình (Training)
Lệnh dưới đây sẽ khởi chạy quá trình huấn luyện phân tán (Distributed Training) cho TransBTS trên dữ liệu BraTS:

```bash
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 20003 train.py
```
*   `--nproc_per_node`: Số lượng GPU sử dụng (ví dụ: 4).
*   `--master_port`: Cổng giao tiếp cho process group.

### 3. Kiểm thử & Đánh giá (Testing)
Để thực hiện kiểm thử với mô hình đã được huấn luyện:

```bash
python3 test.py
```
Sau khi tiến trình kết thúc, file submission có thể được nộp lên trang chủ [BraTS Challenge](https://ipp.cbica.upenn.edu/) để lấy kết quả Dice score chính thức.

---

## 📜 Trích dẫn & Bản quyền (Citation & License)
Dự án này tuân theo giấy phép [Apache 2.0](./LICENSE).
Nếu bạn sử dụng mã nguồn hoặc ý tưởng từ TransBTS/TransBTSV2, vui lòng trích dẫn các bài báo gốc dưới đây để tôn trọng tác giả:

**TransBTS (MICCAI 2021)**:
```bibtex
@inproceedings{wang2021transbts,
  title={TransBTS: Multimodal Brain Tumor Segmentation Using Transformer},
  author={Wang, Wenxuan and Chen, Chen and Ding, Meng and Yu, Hong and Zha, Sen and Li, Jiangyun},
  booktitle={MICCAI 2021: 24th International Conference},
  pages={109--119},
  year={2021},
  organization={Springer}
}
```

**TransBTSV2 (arXiv)**:
```bibtex
@article{li2022transbtsv2,
  title={TransBTSV2: Wider Instead of Deeper Transformer for Medical Image Segmentation},
  author={Li, Jiangyun and Wang, Wenxuan and Chen, Chen and Zhang, Tianxiang and Zha, Sen and Yu, Hong and Wang, Jing},
  journal={arXiv preprint arXiv:2201.12785},
  year={2022}
}
```

---
*Reference implementations*:
*   [setr-pytorch](https://github.com/gupta-abhay/setr-pytorch)
*   [BraTS2017](https://github.com/MIC-DKFZ/BraTS2017)
