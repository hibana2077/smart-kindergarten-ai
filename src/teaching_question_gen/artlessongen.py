from typing import List, Dict
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

class ArtLessonGenerator:
    def __init__(self, model_path: str):
        """初始化美術教案生成器
        
        Args:
            model_path: 微調後的模型路徑
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
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
"""
        }

    def prepare_training_data(self, raw_lessons: List[Dict]) -> pd.DataFrame:
        """準備訓練資料
        
        Args:
            raw_lessons: 原始教案資料列表
            
        Returns:
            處理後的訓練資料DataFrame
        """
        processed_data = []
        
        for lesson in raw_lessons:
            # 將教案轉換為結構化格式
            processed = {
                'input_text': self._create_prompt(lesson),
                'target_text': self._format_lesson(lesson)
            }
            processed_data.append(processed)
            
        return pd.DataFrame(processed_data)

    def _create_prompt(self, lesson: Dict) -> str:
        """根據教案類型生成對應的提示
        
        Args:
            lesson: 教案資料字典
            
        Returns:
            格式化後的提示文字
        """
        template = self.templates[lesson['type']]
        return template.format(
            theme=lesson['theme'],
            age=lesson['age'],
            duration=lesson['duration']
        )

    def _format_lesson(self, lesson: Dict) -> str:
        """將教案轉換為統一格式
        
        Args:
            lesson: 教案資料字典
            
        Returns:
            格式化後的教案文字
        """
        return f"""教學目標：
{lesson['objectives']}

所需材料：
{lesson['materials']}

教學步驟：
{lesson['steps']}

評量方式：
{lesson['evaluation']}
"""

    def generate_lesson(self, lesson_type: str, theme: str, age: str, duration: int) -> str:
        """生成新的教案
        
        Args:
            lesson_type: 教案類型 ('geometry' 或 'color')
            theme: 教學主題
            age: 適用年齡層
            duration: 課程時長(分鐘)
            
        Returns:
            生成的教案文字
        """
        prompt = self.templates[lesson_type].format(
            theme=theme,
            age=age,
            duration=duration
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(inputs["input_ids"],max_length=1000,num_return_sequences=1,temperature=0.7,
            no_repeat_ngram_size=3,
            do_sample=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def save_lesson(self, lesson: str, file_path: str):
        """儲存生成的教案
        
        Args:
            lesson: 教案內容
            file_path: 儲存路徑
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(lesson)

# 使用範例
def main():
    # 初始化生成器
    generator = ArtLessonGenerator("shenzhi-wang/Llama3.1-70B-Chinese-Chat")
    
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