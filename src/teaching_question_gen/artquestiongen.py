from typing import List, Dict
import json
from dataclasses import dataclass
from groq import Groq


@dataclass
class ImageTemplate:
    name: str
    category: str
    svg_template: str
    parameters: Dict[str, str]
    description: str

class ArtTemplateGenerator:
    def __init__(self, groq_api_key: str):
        """初始化模板生成器
        
        Args:
            groq_api_key: Groq API密鑰
        """
        self.client = Groq(api_key=groq_api_key)
        self.templates = self._load_templates()

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
        templates_info = []
        for name, template in self.templates.items():
            param_info = json.dumps(template.parameters, indent=2)
            templates_info.append(f"""- {name}: 
描述: {template.description}
所需參數: {param_info}""")
        
        prompt = f"""基於以下教案內容，請選擇最適合的圖像模板組合，並說明如何調整其參數：

教案內容：
{lesson_content}

可用的模板和其參數要求：
{"".join(templates_info)}

    Please provide the following information in JSON format:
    {{
        "templates": ["template_name1", "template_name2"],
        "parameters": {{
            "template_name1": {{
                // 請使用上述列出的模板所需參數
                // 如果未指定某參數，將使用預設值
            }},
            "template_name2": {{
                // 請使用上述列出的模板所需參數
                // 如果未指定某參數，將使用預設值
            }}
        }},
        "reasons": ["reason1", "reason2"]
    }}

    注意：每個模板的參數必須完全符合其要求的參數名稱。"""

        return prompt

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
        # 使用Groq API進行推論
        prompt = self._get_llm_prompt(lesson_content)
        completion = self.client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            response_format={"type": "json_object"},
            stream=False
        )
        
        # 解析回應
        try:
            template_choices = json.loads(completion.choices[0].message.content)
        except json.JSONDecodeError:
            return []
        
        # 生成圖像
        generated_images = []
        for template_name in template_choices.get("templates", []):
            if template_name in self.templates:
                template = self.templates[template_name]
                # 獲取LLM建議的參數
                llm_params = template_choices.get("parameters", {}).get(template_name, {})
                
                # 合併預設參數和LLM建議的參數
                merged_parameters = template.parameters.copy()  # 先複製預設參數
                merged_parameters.update(llm_params)  # 再更新LLM建議的參數
                
                try:
                    svg = template.svg_template.format(**merged_parameters)
                    generated_images.append({
                        "name": template_name,
                        "svg": svg,
                        "parameters": merged_parameters,
                        "reason": next((reason for reason in template_choices.get("reasons", []) 
                                    if template_name in reason.lower()), "")
                    })
                except KeyError as e:
                    print(f"Warning: Missing parameter {e} for template {template_name}")
                    continue
                except Exception as e:
                    print(f"Error generating SVG for template {template_name}: {e}")
                    continue
                    
        return generated_images

def main():
    # 初始化生成器（請替換為您的API密鑰）
    generator = ArtTemplateGenerator("")
    
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
    import os

    # 輸出結果
    # Ensure the output directory exists
    os.makedirs("output", exist_ok=True)
    
    for image in images:
        print(f"Generated {image['name']}:")
        print(image['svg'])
        print("Parameters:", image['parameters'])
        print("Reason:", image['reason'])
        print("-" * 50)

    # convert svgs to one image
    from datetime import datetime
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF, renderPM
    from PyPDF2 import PdfMerger

    pdf_merger = PdfMerger()
    for i, image in enumerate(images):
        svg_file = f"output/{image['name']}.svg"
        pdf_file = f"output/{image['name']}.pdf"
        with open(svg_file, 'w') as f:
            f.write(image['svg'])
        drawing = svg2rlg(svg_file)
        renderPDF.drawToFile(drawing, pdf_file)
        pdf_merger.append(pdf_file)
        print(f"Saved {image['name']} as {pdf_file}")

    pdf_merger.write(f"output/{datetime.now().strftime('%Y%m%d%H%M%S')}_output.pdf")
    pdf_merger.close()

if __name__ == "__main__":
    main()