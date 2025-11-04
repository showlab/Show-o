import json
import os
from pathlib import Path
from tqdm import tqdm


def download_textvqa_full(output_dir=None):
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "data" / "textvqa"
    else:
        output_dir = Path(output_dir)
    
    print("=" * 60)
    print("TextVQA Dataset Download (10000 семплов)")
    print("=" * 60)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    try:
        from datasets import load_dataset
        print("📦 Загрузка TextVQA через Hugging Face...")
        print("   Загрузка train split...")
        dataset = load_dataset("textvqa", split="train")
        
        max_samples = 10000
        total_samples = len(dataset)
        samples_to_use = min(max_samples, total_samples)
        print(f"   Используем {samples_to_use} примеров из {total_samples} (ограничение: {max_samples})")
        
        data_list = []
        print("📝 Обработка данных...")
        
        seen_images = {}
        dataset_subset = dataset.select(range(samples_to_use))
        for idx, item in enumerate(tqdm(dataset_subset, desc="Processing", total=samples_to_use)):
            image = item['image']
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            if 'image_id' in item:
                image_key = f"textvqa_{item['image_id']}"
            else:
                image_key = f"textvqa_{idx:06d}"
            
            if image_key not in seen_images:
                image_filename = f"{image_key}.png"
                image_path = images_dir / image_filename
                image.save(image_path)
                seen_images[image_key] = image_filename
            
            image_filename = seen_images[image_key]
            answers = item.get('answers', [])
            if isinstance(answers, list):
                answer_texts = answers[:5]  # Топ-5
                answer = answers[0] if len(answers) > 0 else ''
            else:
                answer = str(answers) if answers else ''
                answer_texts = [answer]
            
            data_item = {
                "image": image_filename,
                "question": item.get('question', ''),
                "answers": answer_texts,
                "answer": answer
            }
            
            data_list.append(data_item)
        train_json_path = output_dir / "train.json"
        with open(train_json_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
        
        print(f"✅ TextVQA загружен: {len(data_list)} примеров")
        print(f"   Уникальных изображений: {len(seen_images)}")
        print(f"   JSON: {train_json_path}")
        print(f"   Images: {images_dir}")
        return True
        
    except ImportError:
        print("⚠️  Hugging Face datasets не установлен. Установите: pip install datasets")
        return False
    except Exception as e:
        print(f"❌ Ошибка при загрузке: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_vqav2_full(output_dir=None):
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "data" / "vqav2"
    else:
        output_dir = Path(output_dir)
    
    print("=" * 60)
    print("VQAv2 Dataset Download (10000 семплов)")
    print("=" * 60)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    try:
        from datasets import load_dataset
        print("📦 Загрузка VQAv2 через Hugging Face...")
        dataset = load_dataset("HuggingFaceM4/VQAv2", split="train")
        max_samples = 10000
        total_samples = len(dataset)
        samples_to_use = min(max_samples, total_samples)
        print(f"   Используем {samples_to_use} примеров из {total_samples} (ограничение: {max_samples})")
        
        data_list = []
        print("📝 Обработка данных...")
        
        seen_images = {}
        
        dataset_subset = dataset.select(range(samples_to_use))
        for idx, item in enumerate(tqdm(dataset_subset, desc="Processing", total=samples_to_use)):
            image = item['image']
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            if 'image_id' in item:
                image_key = f"vqav2_{item['image_id']}"
            else:
                image_key = f"vqav2_{idx:06d}"
            
            if image_key not in seen_images:
                image_filename = f"{image_key}.png"
                image_path = images_dir / image_filename
                image.save(image_path)
                seen_images[image_key] = image_filename
            
            image_filename = seen_images[image_key]
            
            answers = item.get('answers', [])
            if isinstance(answers, list):
                answer_texts = answers[:5]
                answer = answers[0] if len(answers) > 0 else ''
            else:
                answer = str(answers) if answers else ''
                answer_texts = [answer]
            
            data_item = {
                "image": image_filename,
                "question": item.get('question', ''),
                "answers": answer_texts,
                "answer": answer
            }
            
            data_list.append(data_item)
        
        train_json_path = output_dir / "train.json"
        with open(train_json_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
        
        print(f"✅ VQAv2 загружен: {len(data_list)} примеров")
        print(f"   Уникальных изображений: {len(seen_images)}")
        print(f"   JSON: {train_json_path}")
        print(f"   Images: {images_dir}")
        return True
        
    except ImportError:
        print("⚠️  Hugging Face datasets не установлен. Установите: pip install datasets")
        return False
    except Exception as e:
        print(f"❌ Ошибка при загрузке: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_docvqa_full(output_dir=None):
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "data" / "docvqa"
    else:
        output_dir = Path(output_dir)
    
    print("=" * 60)
    print("DocVQA Dataset Download (1000 семплов)")
    print("=" * 60)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    try:
        from datasets import load_dataset
        print("📦 Загрузка DocVQA через Hugging Face...")
        
        dataset_names = [
            "ashraq/docvqa",
            "docvqa",
            "allenai/docvqa",
        ]
        
        dataset = None
        for name in dataset_names:
            try:
                print(f"   Пробуем {name}...")
                dataset = load_dataset(name, split="train")
                print(f"   ✅ Успешно загружен через {name}")
                break
            except Exception as e:
                print(f"   ❌ {name}: {str(e)[:100]}")
                continue
        
        if dataset is None:
            raise ValueError("DocVQA не найден на Hugging Face. Попробуйте использовать OKVQA или GQA вместо DocVQA.")
        
        max_samples = 1000
        total_samples = len(dataset)
        samples_to_use = min(max_samples, total_samples)
        print(f"   Используем {samples_to_use} примеров из {total_samples} (ограничение: {max_samples})")
        
        data_list = []
        print("📝 Обработка данных...")
        
        seen_images = {}
        dataset_subset = dataset.select(range(samples_to_use))
        for idx, item in enumerate(tqdm(dataset_subset, desc="Processing", total=samples_to_use)):
            image = item['image']
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            if 'image_id' in item:
                image_key = f"docvqa_{item['image_id']}"
            else:
                image_key = f"docvqa_{idx:06d}"
            
            if image_key not in seen_images:
                image_filename = f"{image_key}.png"
                image_path = images_dir / image_filename
                image.save(image_path)
                seen_images[image_key] = image_filename
            
            image_filename = seen_images[image_key]
            
            # DocVQA формат
            question = item.get('question', '')
            answers = item.get('answers', [])
            if isinstance(answers, list) and len(answers) > 0:
                answer = str(answers[0]) if answers[0] else ''
            else:
                answer = str(answers) if answers else ''
            
            data_item = {
                "image": image_filename,
                "question": str(question),
                "answers": [str(a) for a in answers] if isinstance(answers, list) else [str(answers)],
                "answer": str(answer)
            }
            
            data_list.append(data_item)
        
        train_json_path = output_dir / "train.json"
        with open(train_json_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
        
        print(f"✅ DocVQA загружен: {len(data_list)} примеров")
        print(f"   Уникальных изображений: {len(seen_images)}")
        print(f"   JSON: {train_json_path}")
        print(f"   Images: {images_dir}")
        return True
        
    except ImportError:
        print("⚠️  Hugging Face datasets не установлен. Установите: pip install datasets")
        return False
    except Exception as e:
        print(f"❌ Ошибка при загрузке: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_clevr_full(output_dir=None):
    """
    Загружает CLEVR датасет (10000 семплов)
    
    CLEVR скачивается напрямую с официального сайта: https://cs.stanford.edu/people/jcjohns/clevr/
    """
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "data" / "clevr"
    else:
        output_dir = Path(output_dir)
    
    print("=" * 60)
    print("CLEVR Dataset Download (10000 семплов)")
    print("=" * 60)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    try:
        import requests
        import zipfile
        import tempfile
        import shutil
        
        print("📦 Загрузка CLEVR напрямую...")
        
        # URL для скачивания CLEVR (официальный сайт)
        # Используем прямую ссылку на данные
        clevr_urls = [
            "https://cs.stanford.edu/people/jcjohns/clevr/CLEVR_v1.0.zip",
            "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip",
        ]
        
        # Создаем временную директорию для загрузки
        temp_dir = tempfile.mkdtemp()
        zip_path = Path(temp_dir) / "CLEVR_v1.0.zip"
        
        downloaded = False
        for url in clevr_urls:
            try:
                print(f"   Пробуем скачать с: {url}")
                print("   Это может занять время (датасет ~20GB)...")
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                print(f"   Размер файла: {total_size / (1024**3):.2f} GB")
                
                with open(zip_path, 'wb') as f:
                    for chunk in tqdm(response.iter_content(chunk_size=8192), 
                                     total=total_size // 8192, 
                                     desc="   Скачивание", 
                                     unit="KB"):
                        f.write(chunk)
                
                print("   ✅ Файл скачан")
                downloaded = True
                break
            except Exception as e:
                print(f"   ❌ Ошибка: {str(e)[:100]}")
                continue
        
        if not downloaded:
            print("\n⚠️  Не удалось скачать CLEVR автоматически.")
            print("   Попробуйте скачать вручную:")
            print("   1. Перейдите на https://cs.stanford.edu/people/jcjohns/clevr/")
            print("   2. Скачайте CLEVR_v1.0.zip")
            print("   3. Распакуйте и используйте скрипт для обработки")
            return False
        
        # Распаковываем ZIP
        print("   Распаковка архива...")
        extract_dir = Path(temp_dir) / "clevr_extracted"
        extract_dir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Ищем файлы с вопросами и изображениями
        questions_file = None
        images_source_dir = None
        
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if 'questions' in file and file.endswith('.json'):
                    questions_file = Path(root) / file
                if 'train' in dirs:
                    images_source_dir = Path(root) / 'images' / 'train'
                    break
            if questions_file and images_source_dir:
                break
        
        if not questions_file or not images_source_dir:
            # Попробуем найти в стандартной структуре
            questions_file = extract_dir / "CLEVR_v1.0" / "questions" / "CLEVR_train_questions.json"
            images_source_dir = extract_dir / "CLEVR_v1.0" / "images" / "train"
        
        if not questions_file.exists() or not images_source_dir.exists():
            print(f"   ❌ Не найдены файлы:")
            print(f"      Questions: {questions_file}")
            print(f"      Images: {images_source_dir}")
            return False
        
        print(f"   ✅ Найдены файлы:")
        print(f"      Questions: {questions_file}")
        print(f"      Images: {images_source_dir}")
        
        # Загружаем вопросы
        with open(questions_file, 'r') as f:
            questions_data = json.load(f)
        
        questions = questions_data.get('questions', [])
        print(f"   Всего вопросов: {len(questions)}")
        
        # Ограничиваем до 10000 семплов
        max_samples = 10000
        samples_to_use = min(max_samples, len(questions))
        print(f"   Используем {samples_to_use} примеров (ограничение: {max_samples})")
        
        # Получаем список изображений
        image_files = list(images_source_dir.glob("*.png"))
        image_dict = {img.stem: img for img in image_files}
        
        data_list = []
        print("📝 Обработка данных...")
        
        seen_images = {}
        
        for idx, item in enumerate(tqdm(questions[:samples_to_use], desc="Processing", total=samples_to_use)):
            image_filename_hf = item.get('image_filename', '')
            image_id = item.get('image_index', idx)
            
            # Ищем изображение
            image_path = None
            if image_filename_hf:
                image_path = images_source_dir / image_filename_hf
            elif str(image_id) in image_dict:
                image_path = image_dict[str(image_id)]
            else:
                # Пробуем найти по паттерну
                for img_file in image_files:
                    if str(image_id) in img_file.stem or image_filename_hf in img_file.name:
                        image_path = img_file
                        break
            
            if not image_path or not image_path.exists():
                continue
            
            # Копируем изображение
            image_key = f"clevr_{image_id}"
            if image_key not in seen_images:
                dest_image_path = images_dir / f"{image_key}.png"
                shutil.copy2(image_path, dest_image_path)
                seen_images[image_key] = f"{image_key}.png"
            
            image_filename = seen_images[image_key]
            
            # Формируем запись
            question = item.get('question', '')
            answer = item.get('answer', '')
            
            data_item = {
                "image": image_filename,
                "question": str(question),
                "answers": [str(answer)] if answer else [],
                "answer": str(answer)
            }
            
            data_list.append(data_item)
        
        # Сохраняем JSON
        train_json_path = output_dir / "train.json"
        with open(train_json_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
        
        # Очищаем временные файлы
        shutil.rmtree(temp_dir)
        
        print(f"✅ CLEVR загружен: {len(data_list)} примеров")
        print(f"   Уникальных изображений: {len(seen_images)}")
        print(f"   JSON: {train_json_path}")
        print(f"   Images: {images_dir}")
        return True
        
    except ImportError:
        print("⚠️  Нужны библиотеки: requests. Установите: pip install requests")
        return False
    except Exception as e:
        print(f"❌ Ошибка при загрузке: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Загрузка датасетов для экспериментов MoE (по 10000 семплов каждый)')
    parser.add_argument('--textvqa', action='store_true', help='Загрузить TextVQA (10000 семплов)')
    parser.add_argument('--vqav2', action='store_true', help='Загрузить VQAv2 (10000 семплов)')
    parser.add_argument('--docvqa', action='store_true', help='Загрузить DocVQA (10000 семплов) - документы')
    parser.add_argument('--clevr', action='store_true', help='Загрузить CLEVR (10000 семплов) - синтетические 3D сцены')
    parser.add_argument('--all', action='store_true', help='Загрузить все доступные датасеты (по 10000 семплов каждый)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Базовый путь для сохранения (по умолчанию: moe_experiments/data относительно скрипта)')
    
    args = parser.parse_args()
    
    success_count = 0
    
    # Определяем базовый путь
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        script_dir = Path(__file__).parent
        base_output_dir = script_dir / "data"
    
    if args.all or args.textvqa:
        if args.output_dir:
            textvqa_output = base_output_dir / "textvqa"
        else:
            textvqa_output = None
        if download_textvqa_full(textvqa_output):
            success_count += 1
    
    if args.all or args.vqav2:
        if args.output_dir:
            vqav2_output = base_output_dir / "vqav2"
        else:
            vqav2_output = None
        if download_vqav2_full(vqav2_output):
            success_count += 1
    
    if args.all or args.docvqa:
        if args.output_dir:
            docvqa_output = base_output_dir / "docvqa"
        else:
            docvqa_output = None
        if download_docvqa_full(docvqa_output):
            success_count += 1
    
    if args.all or args.clevr:
        if args.output_dir:
            clevr_output = base_output_dir / "clevr"
        else:
            clevr_output = None
        if download_clevr_full(clevr_output):
            success_count += 1
    
    if success_count == 0:
        print("\n⚠️  Не указаны датасеты для загрузки.")
        print("Используйте --textvqa, --vqav2, --docvqa, --clevr или --all")
        print("\nПримеры:")
        print("  python download_full_datasets.py --docvqa")
        print("  python download_full_datasets.py --clevr")
        print("  python download_full_datasets.py --docvqa --clevr  # Для двух разных доменов")
    else:
        print(f"\n✅ Успешно загружено датасетов: {success_count}")


if __name__ == "__main__":
    main()

