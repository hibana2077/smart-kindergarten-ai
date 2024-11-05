from typing import List, Dict
import json
from groq import Groq
import pandas as pd

class ArtLessonGenerator:
    def __init__(self, groq_api_key: str):
        """初始化美術教案生成器
        
        Args:
            groq_api_key: Groq API密鑰
        """
        self.client = Groq(api_key=groq_api_key)
        
        # 預設提示模板
        self.templates = {
            "geometry": """生成一份幼兒園幾何圖形教案:
主題: {theme}
年齡層: {age}
時長: {duration}分鐘
請包含:
1. 教學目標
2. 所需材料
3. 教學步驟
4. 評量方式

please use JSON format to answer, including the following fields:
    "objectives": ["教學目標1", "教學目標2"...],
    "materials": ["材料1", "材料2"...],
    "steps": ["步驟1", "步驟2"...],
    "evaluation": ["評量方式1", "評量方式2"...]
""",
            "color": """生成一份幼兒園色彩繪畫教案:
主題: {theme}
年齡層: {age}
時長: {duration}分鐘
請包含:
1. 教學目標
2. 所需材料
3. 教學步驟
4. 評量方式

please use JSON format to answer, including the following fields:
    'objectives': ["教學目標1", "教學目標2"...],
    'materials': ["材料1", "材料2"...],
    'steps': ["步驟1", "步驟2"...],
    'evaluation': ["評量方式1", "評量方式2"...]
"""
        }

    def generate_lesson(self, lesson_type: str, theme: str, age: str, duration: int) -> Dict:
        """生成新的教案
        
        Args:
            lesson_type: 教案類型 ('geometry' 或 'color')
            theme: 教學主題
            age: 適用年齡層
            duration: 課程時長(分鐘)
            
        Returns:
            生成的教案字典
        """
        print(f"Generating {lesson_type} lesson on '{theme}' for {age} students ({duration} minutes)")
        prompt = self.templates[lesson_type].format(
            theme=theme,
            age=age,
            duration=duration
        )
        
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
        
        return json.loads(completion.choices[0].message.content)

    def _format_lesson(self, lesson_dict: Dict) -> str:
        """將教案字典格式化為文字
        
        Args:
            lesson_dict: 教案字典
            
        Returns:
            格式化後的教案文字
        """
        return f"""教學目標：
{''.join(f'- {obj}\n' for obj in lesson_dict['objectives'])}

所需材料：
{''.join(f'- {mat}\n' for mat in lesson_dict['materials'])}

教學步驟：
{''.join(f'{i+1}. {step}\n' for i, step in enumerate(lesson_dict['steps']))}

評量方式：
{''.join(f'- {eval}\n' for eval in lesson_dict['evaluation'])}
"""

    def save_lesson(self, lesson: Dict, file_path: str):
        """儲存生成的教案
        
        Args:
            lesson: 教案字典
            file_path: 儲存路徑
        """
        formatted_lesson = self._format_lesson(lesson)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(formatted_lesson)

# 使用範例
def main():
    # 初始化生成器（請替換為您的API密鑰）
    generator = ArtLessonGenerator("")
    
    # 生成幾何圖形教案
    geometry_lesson = generator.generate_lesson(
        lesson_type="geometry",
        theme="認識圓形和方形",
        age="4-5歲",
        duration=30
    )
    
    # 生成色彩教案
    color_lesson = generator.generate_lesson(
        lesson_type="color",
        theme="春天的顏色",
        age="5-6歲",
        duration=45
    )
    
    # 儲存教案
    generator.save_lesson(geometry_lesson, "geometry_lesson.txt")
    generator.save_lesson(color_lesson, "color_lesson.txt")

if __name__ == "__main__":
    main()