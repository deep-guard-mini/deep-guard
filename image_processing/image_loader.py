from PIL import Image


def load_image(image_path):
    image = Image.open(image_path)
    return image.convert("RGB")

if __name__ == "__main__":
    image = load_image("test_images/test.jpeg")

    print("Image loaded successfully")
    print("Size:", image.size)
    print("Mode:", image.mode)