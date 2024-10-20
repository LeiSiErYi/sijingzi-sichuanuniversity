from pdf2image import convert_from_path
from paddleocr import PaddleOCR
import cv2
import numpy as np
import os
from PIL import Image  

class PDFTextRecognizer:
    def __init__(self, lang='ch'):
        """
        初始化 OCR 模型，并设置语言类型
        :param lang: OCR语言类型，默认为中文 'ch'
        """
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)  # 初始化OCR模型

    def pdf_to_images(self, pdf_path):
        """
        将PDF每页转换为图片
        :param pdf_path: PDF 文件路径
        :return: 转换后的图片列表
        """
        return convert_from_path(pdf_path)

    def ocr_and_draw(self, image_path):
        """
        对单张图片进行 OCR 识别，并绘制识别结果（每个字符都框出）
        :param image_path: 输入图片的文件路径
        :return: 带有识别框的图片
        """
        # 使用 OpenCV 从路径读取图像
        image_cv = cv2.imread(image_path)  # 直接读取图像文件

        # OCR 识别
        result = self.ocr.ocr(image_cv, cls=True)

        # 提取识别的边框、文本和分数
        for line in result[0]:
            box = line[0]  # 检测到的整列文本框
            text = line[1][0]  # 检测到的文本内容
            score = line[1][1]  # 置信度分数
            # print(text)
            # 计算每个字符的高度
            char_height = (box[2][1] - box[0][1]) / len(text)  # 每个字符的高度

            for i, char in enumerate(text):
                # 计算每个字符的框坐标，基于字符高度进行分隔
                char_box = [
                    [box[0][0], box[0][1] + i * char_height+7],
                    [box[1][0], box[0][1] + i * char_height+7],
                    [box[1][0], box[0][1] + (i + 1) * char_height-7],
                    [box[0][0], box[0][1] + (i + 1) * char_height-7]
                ]

                # 绘制每个字符的框
                pts = np.array(char_box, dtype=np.int32)
                cv2.polylines(image_cv, [pts], isClosed=True, color=(0, 255, 0), thickness=5)  # 绘制边框

                # # 计算文本位置并绘制字符
                # x, y = int(pts[0][0]), int(pts[0][1] - 5)  # 文本位置稍微上移
                # # cv2.putText(image_cv, char, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # 绘制字符

        return image_cv

    def process_pdf(self, pdf_path, output_folder):
        """
        主函数：处理整个PDF，进行OCR识别并保存结果
        :param pdf_path: PDF 文件路径
        :param output_folder: 保存识别结果图片的文件夹
        """
        # 创建输出文件夹（如果不存在）
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Step 1: 将 PDF 转为图片列表
        images = self.pdf_to_images(pdf_path)

        # 用于保存结果图片路径
        result_images = []

        # Step 2: 对每页图片进行 OCR 识别并绘制结果
        for idx, image in enumerate(images):
            temp_image_path = f'temp_image_{idx}.png'  # 临时保存转换的图片
            image.save(temp_image_path)  # 保存为临时文件
            result_image = self.ocr_and_draw(temp_image_path)

            # Step 3: 保存带有识别框的图片
            output_path = os.path.join(output_folder, f'result_page_{idx + 1}.png')
            cv2.imwrite(output_path, result_image)  # 使用 OpenCV 保存图像
            print(f'Page {idx + 1} processed and saved as {output_path}')

            # 保存结果图片路径
            result_images.append(output_path)

            os.remove(temp_image_path)  # 删除临时文件

        # 合并结果图片为PDF
        self.images_to_pdf(result_images, os.path.join(output_folder, 'combined_results.pdf'))

    def images_to_pdf(self, image_paths, output_pdf_path):
        """
        将图片合并为PDF文件
        :param image_paths: 图片文件路径列表
        :param output_pdf_path: 输出PDF文件路径
        """
        images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
        if images:
            images[0].save(output_pdf_path, save_all=True, append_images=images[1:])
            print(f'Result images merged into PDF and saved as {output_pdf_path}')


# 示例调用
if __name__ == "__main__":
    pdf_path = 'an.pdf'  # 替换为你的PDF文件路径
    output_folder = 'results'  # 结果图片保存的文件夹路径
    # 创建 PDF 处理器对象
    recognizer = PDFTextRecognizer()
    # 处理PDF并保存结果
    recognizer.process_pdf(pdf_path, output_folder)
