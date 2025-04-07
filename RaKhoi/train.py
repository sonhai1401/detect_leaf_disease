import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
from tqdm import tqdm  # Tiến trình hiển thị quá trình train
from google.colab import drive  # Import Google Drive

# Kết nối Google Drive để lưu dữ liệu và mô hình
drive.mount('/content/drive')

def get_data_loaders(data_dir, batch_size=32, num_workers=2):
    transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.RandomHorizontalFlip(),  # Lật ảnh ngang
      transforms.RandomRotation(20),  # Xoay ngẫu nhiên ±20 độ
      transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Điều chỉnh độ sáng & tương phản
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])


    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "valid"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, len(train_dataset.classes)

# Hàm tạo mô hình MobileNetV2
def create_model(num_classes):
    model = models.mobilenet_v2(pretrained=True)  # Tải mô hình MobileNetV2 đã được huấn luyện trước
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)  # Thay đổi lớp đầu ra phù hợp với số lớp
    return model

# Hàm huấn luyện mô hình
def train_model(model, train_loader, criterion, optimizer, device, epochs=100):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        start_time = time.time()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}, Accuracy: {correct/total:.4f}, Time: {epoch_time:.2f}s")


# Hàm đánh giá mô hình
def evaluate_model(model, val_loader, device):
    model.to(device)
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    print(f"Validation Accuracy: {correct / total:.4f}")

# Hàm chính để thực thi toàn bộ pipeline
def main():
    data_dir = "/content/dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"  # Cập nhật đường dẫn thư mục dữ liệu trên Google Drive
    model_path = "/content/drive/MyDrive/plant_disease_mobilenetv2_2.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tải dữ liệu
    train_loader, val_loader, num_classes = get_data_loaders(data_dir)

    # Khởi tạo mô hình
    model = create_model(num_classes)

    # Nếu đã có mô hình được lưu, tải lại để tiếp tục training
    if os.path.exists(model_path):
        print("Đang tải mô hình đã lưu...")
        model.load_state_dict(torch.load(model_path))
        print("Mô hình được tải thành công!")

    # Cấu hình hàm mất mát và bộ tối ưu
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Huấn luyện mô hình
    train_model(model, train_loader, criterion, optimizer, device)

    # Đánh giá mô hình
    evaluate_model(model, val_loader, device)

    # Lưu mô hình sau khi huấn luyện vào Google Drive
    torch.save(model.state_dict(), model_path)
    print(f"Model saved successfully at {model_path}!")

# Chạy chương trình nếu file này được thực thi
if __name__ == "__main__":
    main()
