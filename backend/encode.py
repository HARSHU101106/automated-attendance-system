import face_recognition
import os
import pickle

print("🔄 Encoding started...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(BASE_DIR, "..", "dataset")
encoding_dir = os.path.join(BASE_DIR, "..", "encodings")
encoding_file = os.path.join(encoding_dir, "encodings.pkl")

encodings = []
names = []

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        print(f"Processing {person}/{img_name}...")

        image = face_recognition.load_image_file(img_path)
        face_enc = face_recognition.face_encodings(image)

        if face_enc:
            encodings.append(face_enc[0])
            names.append(person)
        else:
            print(f"❌ No face found in {img_name}")

data = {"encodings": encodings, "names": names}

# ✅ Create encodings folder in ROOT (not backend)
os.makedirs(encoding_dir, exist_ok=True)

with open(encoding_file, "wb") as f:
    pickle.dump(data, f)

print("✅ Encoding completed!")
print(f"📁 Saved at: {encoding_file}")