from ultralytics import YOLO

# Load model
model = YOLO("model/best.pt")

# Test 1 gambar
results = model("test_images/test.jpg", conf=0.2)

# Print hasil
print("Class names:", results[0].names)
print("Boxes:", results[0].boxes)

# Simpan hasil gambar
results[0].save(filename="hasil.jpg")