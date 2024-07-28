import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


# 横向合并图片
def merge_images_horizontally(images):
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    new_image = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        new_image.paste(img, (x_offset, 0))
        x_offset += img.width
    return new_image


# 纵向合并图片
def merge_images_vertically(images):
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    total_height = sum(heights)

    new_image = Image.new('RGB', (max_width, total_height))
    y_offset = 0
    for img in images:
        new_image.paste(img, (0, y_offset))
        y_offset += img.height
    return new_image


def add_labels(image, x_labels, y_labels, label_width=150, label_height=100, font_size=30):
    # 创建一个额外的空间以添加纵坐标标签
    img_with_labels = Image.new('RGB', (image.width + label_width, image.height + label_height), (255, 255, 255))

    # 粘贴原始图像到新图像
    img_with_labels.paste(image, (label_width, label_height))

    draw = ImageDraw.Draw(img_with_labels)

    # 加载字体
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # 添加横坐标标签（居中对齐）
    x_label_width = image.width // len(x_labels)
    for i, label in enumerate(x_labels):
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x_position = label_width + (i * x_label_width) + (x_label_width - text_width) // 2
        y_position = label_height // 2 - text_height // 2
        draw.text((x_position, y_position), label, fill="black", font=font)

    # 添加纵坐标标签（居中对齐）
    y_label_height = image.height // len(y_labels)
    for i, label in enumerate(y_labels):
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x_position = 10
        y_position = label_height + (i * y_label_height) + (y_label_height - text_height) // 2
        draw.text((x_position, y_position), label, fill="black", font=font)

    return img_with_labels


def main(input_folder, output_file):
    subfolders = [f.path for f in os.scandir(input_folder) if f.is_dir()]

    # 对子文件夹进行排序
    subfolders.sort(key=lambda folder: int(os.path.basename(folder).split('-')[-1]))

    combined_images = []
    x_labels = [f'cartoon_{i}.png' for i in range(1, 6)]

    # 处理子文件夹名称，去掉前缀
    y_labels = [os.path.basename(folder).replace('output_file_checkpoint-', '') for folder in subfolders]

    for folder in subfolders:
        images = []
        for i in range(1, 6):
            img_path = os.path.join(folder, f'cartoon_{i}.png')
            img = Image.open(img_path)
            images.append(img)

        combined_img = merge_images_horizontally(images)
        combined_images.append(combined_img)

    final_image = merge_images_vertically(combined_images)

    # 添加坐标标签
    final_image_with_labels = add_labels(final_image, x_labels, y_labels, label_width=150, label_height=100,
                                         font_size=50)

    final_image_with_labels.save(output_file)


# 使用示例
input_folder = './output_file'
output_file = './output_file/final_combined_image_with_labels.png'
main(input_folder, output_file)

# 可选：显示结果
img = Image.open(output_file)
plt.imshow(img)
plt.axis('off')
plt.show()
