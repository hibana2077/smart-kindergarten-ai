from typing import List, Dict, Optional
import json
from dataclasses import dataclass
import random

@dataclass
class ImageTemplate:
    name: str
    category: str
    svg_template: str
    parameters: Dict[str, str]
    description: str

class ArtTemplateGenerator:
    def __init__(self, llm_model_path: str):
        """初始化模板生成器
        
        Args:
            llm_model_path: LLM模型路徑
        """
        self.llm = self._init_llm(llm_model_path)
        self.templates = self._load_templates()
        
    def _init_llm(self, model_path: str):
        """初始化LLM模型"""
        # 這裡可以根據需求使用不同的LLM實現
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        return {"model": model, "tokenizer": tokenizer}

    def _load_templates(self) -> Dict[str, ImageTemplate]:
        """載入預定義的SVG模板"""
        return {
            "basic_circle": ImageTemplate(
                name="basic_circle",
                category="geometry",
                svg_template='''
                <svg viewBox="0 0 100 100">
                    <circle cx="{cx}" cy="{cy}" r="{radius}" 
                            fill="{fill_color}" stroke="{stroke_color}" 
                            stroke-width="{stroke_width}"/>
                </svg>
                ''',
                parameters={
                    "cx": "50",
                    "cy": "50",
                    "radius": "40",
                    "fill_color": "#FFE4E1",
                    "stroke_color": "#FF69B4",
                    "stroke_width": "2"
                },
                description="基礎圓形，適合教授圓形概念"
            ),
            "basic_square": ImageTemplate(
                name="basic_square",
                category="geometry",
                svg_template='''
                <svg viewBox="0 0 100 100">
                    <rect x="{x}" y="{y}" width="{width}" height="{height}" 
                          fill="{fill_color}" stroke="{stroke_color}" 
                          stroke-width="{stroke_width}"/>
                </svg>
                ''',
                parameters={
                    "x": "20",
                    "y": "20",
                    "width": "60",
                    "height": "60",
                    "fill_color": "#E6E6FA",
                    "stroke_color": "#9370DB",
                    "stroke_width": "2"
                },
                description="基礎方形，適合教授方形概念"
            ),
            "basic_triangle": ImageTemplate(
                name="basic_triangle",
                category="geometry",
                svg_template='''
                <svg viewBox="0 0 100 100">
                    <path d="M {x1},{y1} L {x2},{y2} L {x3},{y3} Z" 
                          fill="{fill_color}" stroke="{stroke_color}" 
                          stroke-width="{stroke_width}"/>
                </svg>
                ''',
                parameters={
                    "x1": "50", "y1": "20",
                    "x2": "20", "y2": "80",
                    "x3": "80", "y3": "80",
                    "fill_color": "#98FB98",
                    "stroke_color": "#3CB371",
                    "stroke_width": "2"
                },
                description="基礎三角形，適合教授三角形概念"
            ),
            "color_palette": ImageTemplate(
                name="color_palette",
                category="color",
                svg_template='''
                <svg viewBox="0 0 300 100">
                    <rect x="0" y="0" width="60" height="100" fill="{color1}"/>
                    <rect x="60" y="0" width="60" height="100" fill="{color2}"/>
                    <rect x="120" y="0" width="60" height="100" fill="{color3}"/>
                    <rect x="180" y="0" width="60" height="100" fill="{color4}"/>
                    <rect x="240" y="0" width="60" height="100" fill="{color5}"/>
                </svg>
                ''',
                parameters={
                    "color1": "#FF0000",
                    "color2": "#00FF00",
                    "color3": "#0000FF",
                    "color4": "#FFFF00",
                    "color5": "#FF00FF"
                },
                description="顏色調色盤，適合教授色彩概念"
            )
        }

    def _get_llm_prompt(self, lesson_content: str) -> str:
        """生成提示語句讓LLM選擇合適的模板"""
        return f"""基於以下教案內容，請選擇最適合的圖像模板組合，並說明如何調整其參數：

教案內容：
{lesson_content}

可用的模板有：
{self._get_template_descriptions()}

請用JSON格式回答，包含：
1. 選擇的模板列表
2. 每個模板的參數調整
3. 使用理由
"""

    def _get_template_descriptions(self) -> str:
        """獲取所有模板的描述"""
        return "\n".join([
            f"- {name}: {template.description}"
            for name, template in self.templates.items()
        ])

    def _parse_llm_response(self, response: str) -> Dict:
        """解析LLM的回應"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"error": "無法解析LLM回應"}

    def generate_images(self, lesson_content: str) -> List[Dict[str, str]]:
        """根據教案內容生成圖像
        
        Args:
            lesson_content: 教案內容
            
        Returns:
            生成的SVG圖像列表
        """
        # 1. 讓LLM分析教案並選擇模板
        prompt = self._get_llm_prompt(lesson_content)
        inputs = self.llm["tokenizer"](prompt, return_tensors="pt")
        outputs = self.llm["model"].generate(inputs["input_ids"])
        llm_response = self.llm["tokenizer"].decode(outputs[0])
        
        # 2. 解析LLM的建議
        template_choices = self._parse_llm_response(llm_response)
        
        # 3. 生成圖像
        generated_images = []
        for template_name in template_choices.get("templates", []):
            if template_name in self.templates:
                template = self.templates[template_name]
                parameters = template_choices.get("parameters", {}).get(template_name, template.parameters)
                
                svg = template.svg_template.format(**parameters)
                generated_images.append({
                    "name": template_name,
                    "svg": svg,
                    "parameters": parameters
                })
                
        return generated_images

def main():
    # 初始化生成器
    generator = ArtTemplateGenerator("path/to/your/llm/model")
    
    # 測試教案內容
    lesson_content = """
    主題：認識基本形狀
    教學目標：
    1. 能夠識別並說出圓形、方形、三角形
    2. 能夠在日常生活中找出相應的形狀
    
    教學步驟：
    1. 展示各種基本形狀
    2. 讓學生描述形狀特徵
    3. 進行形狀配對遊戲
    """
    
    # 生成圖像
    images = generator.generate_images(lesson_content)
    
    # 輸出結果
    for image in images:
        print(f"Generated {image['name']}:")
        print(image['svg'])
        print("Parameters:", image['parameters'])
        print("-" * 50)

if __name__ == "__main__":
    main()