import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

def describe_image(image):
    # 初始化模型和处理器
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    # 处理输入图像
    inputs = processor(image, return_tensors="pt")

    # 生成图像描述
    outputs = model.generate(**inputs)
    description = processor.decode(outputs[0], skip_special_tokens=True)

    return description

# 创建Gradio界面
iface = gr.Interface(fn=describe_image,
                     inputs=gr.Image(type="pil", label="上传图片"),
                     outputs=gr.Textbox(label="图片描述"),
                     title="图像内容描述工具",
                     description="上传一张图片，模型将生成一个关于图片内容的描述。")

# 运行Gradio应用
if __name__ == "__main__":
    iface.launch()
