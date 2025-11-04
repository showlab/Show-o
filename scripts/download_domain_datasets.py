import os
import json
import requests
from pathlib import Path
from tqdm import tqdm
from PIL import Image


def download_file(url, output_path, description="Downloading"):
    """Скачивает файл с прогресс-баром"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f, tqdm(
        desc=description,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def download_vqav2(output_dir="./data/vqav2"):
    print("=" * 60)
    print("VQAv2 Dataset Download")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    try:
        from datasets import load_dataset
        print("📦 Загрузка VQAv2 через Hugging Face...")
        
        # Загружаем train split (можно использовать часть данных)
        print("   Загрузка train split (может занять время)...")
        dataset = load_dataset("HuggingFaceM4/VQAv2", split="train")
        
        # Ограничиваем количество для быстрого старта (можно убрать для полного датасета)
        max_samples = 50000  # Используем 50k примеров из ~443k
        print(f"   Используем первые {max_samples} примеров (для полного датасета уберите ограничение)")
        
        data_list = []
        print("📝 Обработка данных...")
        
        seen_images = {}  # Чтобы не дублировать изображения
        
        # Используем слайсинг вместо take для совместимости
        dataset_subset = dataset.select(range(min(max_samples, len(dataset))))
        
        for idx, item in enumerate(tqdm(dataset_subset, desc="Processing", total=min(max_samples, len(dataset)))):
            # Сохраняем изображение только один раз
            image = item['image']
            
            # Создаем уникальный ключ для изображения
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
            
            # Формируем запись
            answers = item.get('answers', [])
            if isinstance(answers, list):
                # Берем самые частые ответы
                if len(answers) > 0:
                    if isinstance(answers[0], dict):
                        # Формат: [{"answer": "yes", "answer_confidence": "yes"}, ...]
                        answer_texts = [a.get('answer', '') for a in answers if a.get('answer')]
                        answer = answer_texts[0] if answer_texts else ''
                    else:
                        answer_texts = answers
                        answer = answer_texts[0] if answer_texts else ''
                else:
                    answer = ''
            else:
                answer = str(answers) if answers else ''
                answer_texts = [answer]
            
            data_item = {
                "image": image_filename,
                "question": item.get('question', ''),
                "answers": answer_texts[:5] if isinstance(answer_texts, list) else [answer],  # Топ-5 ответов
                "answer": answer
            }
            
            data_list.append(data_item)
        
        # Сохраняем JSON
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


def download_textvqa(output_dir="./data/textvqa"):
    print("=" * 60)
    print("TextVQA Dataset Download")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    try:
        from datasets import load_dataset
        print("📦 Загрузка TextVQA через Hugging Face...")
        
        # Загружаем train split
        print("   Загрузка train split...")
        dataset = load_dataset("textvqa", split="train")
        
        # Ограничиваем для быстрого старта
        max_samples = 30000  # Используем 30k примеров
        print(f"   Используем первые {max_samples} примеров")
        
        data_list = []
        print("📝 Обработка данных...")
        
        seen_images = {}
        
        # Используем слайсинг вместо take для совместимости
        dataset_subset = dataset.select(range(min(max_samples, len(dataset))))
        
        for idx, item in enumerate(tqdm(dataset_subset, desc="Processing", total=min(max_samples, len(dataset)))):
            # Сохраняем изображение
            image = item['image']
            
            # Создаем уникальный ключ
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
            
            # Формируем запись
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
        
        # Сохраняем JSON
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


def download_clevr(output_dir="./data/clevr"):
    print("=" * 60)
    print("CLEVR Dataset Download")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    try:
        import zipfile
        from datasets import load_dataset
        
        # Попробуем загрузить через Hugging Face сначала (может быть доступно под другим именем)
        print("📦 Попытка загрузки CLEVR через Hugging Face...")
        dataset_names = ["allenai/clevr-dataset", "yujiali/clevr-dataset-gen"]
        
        dataset = None
        for name in dataset_names:
            try:
                print(f"   Пробуем {name}...")
                dataset = load_dataset(name, split="train")
                print(f"   ✅ Успешно загружен через {name}")
                break
            except Exception:
                continue
        
        if dataset is None:
            # Если не получилось через Hugging Face, используем прямую загрузку
            print("   ⚠️  Hugging Face загрузка недоступна, используем прямую загрузку")
            print("\n   Для загрузки CLEVR необходимо:")
            print("   1. Перейти на https://cs.stanford.edu/people/jcjohns/clevr/")
            print("   2. Скачать CLEVR_v1.0.zip")
            print("   3. Распаковать в", output_dir)
            print("   4. Структура должна быть:")
            print("      clevr/")
            print("        images/")
            print("          train/")
            print("            CLEVR_train_*.png")
            print("        questions/")
            print("          CLEVR_train_questions.json")
            print("\n   Или используйте другой датасет (--vizwiz)")
            return False
        
        print(f"   Всего доступно примеров: {len(dataset)}")
        
        data_list = []
        print("📝 Обработка данных...")
        
        seen_images = {}
        
        for idx, item in enumerate(tqdm(dataset, desc="Processing")):
            # Сохраняем изображение
            image = item.get('image')
            if image is None:
                continue
            
            # Создаем уникальный ключ
            if 'image_filename' in item:
                image_key = item['image_filename'].replace('.png', '').replace('.jpg', '')
            elif 'image_id' in item:
                image_key = f"clevr_{item['image_id']}"
            else:
                image_key = f"clevr_{idx:06d}"
            
            if image_key not in seen_images:
                image_filename = f"{image_key}.png"
                image_path = images_dir / image_filename
                image.save(image_path)
                seen_images[image_key] = image_filename
            
            image_filename = seen_images[image_key]
            
            # Формируем запись
            answer = item.get('answer', '')
            if not answer:
                answers = item.get('answers', [])
                if isinstance(answers, list) and len(answers) > 0:
                    answer = answers[0] if isinstance(answers[0], str) else str(answers[0])
                else:
                    answer = str(answers) if answers else ''
            
            answer_texts = [answer] if answer else []
            
            data_item = {
                "image": image_filename,
                "question": item.get('question', ''),
                "answers": answer_texts,
                "answer": answer
            }
            
            if 'question_id' in item:
                data_item['question_id'] = item['question_id']
            
            data_list.append(data_item)
        
        # Сохраняем JSON
        train_json_path = output_dir / "train.json"
        with open(train_json_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
        
        print(f"✅ CLEVR загружен: {len(data_list)} примеров")
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


def download_vizwiz(output_dir="./data/vizwiz"):
    print("=" * 60)
    print("VizWiz Dataset Download")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    try:
        from datasets import load_dataset
        print("📦 Загрузка VizWiz через Hugging Face...")
        
        # Загружаем train split
        print("   Загрузка train split (может занять время)...")
        dataset = load_dataset("vizwiz", split="train")
        
        total_available = len(dataset)
        print(f"   Всего доступно примеров: {total_available}")
        print(f"   Используем все доступные примеры")
        
        data_list = []
        print("📝 Обработка данных...")
        
        seen_images = {}
        
        for idx, item in enumerate(tqdm(dataset, desc="Processing")):
            # Сохраняем изображение
            image = item.get('image')
            if image is None:
                print(f"   ⚠️  Пропущен пример {idx}: нет изображения")
                continue
            
            # Создаем уникальный ключ
            if 'image_id' in item:
                image_key = f"vizwiz_{item['image_id']}"
            elif 'imageId' in item:
                image_key = f"vizwiz_{item['imageId']}"
            else:
                image_key = f"vizwiz_{idx:06d}"
            
            if image_key not in seen_images:
                image_filename = f"{image_key}.png"
                image_path = images_dir / image_filename
                image.save(image_path)
                seen_images[image_key] = image_filename
            
            image_filename = seen_images[image_key]
            
            # Формируем запись
            # VizWiz обычно имеет поле "answer" или "answers"
            answer = item.get('answer', '')
            if not answer:
                answers = item.get('answers', [])
                if isinstance(answers, list) and len(answers) > 0:
                    if isinstance(answers[0], dict):
                        # Может быть формат [{"answer": "...", ...}, ...]
                        answer_texts = [a.get('answer', '') for a in answers if a.get('answer')]
                        answer = answer_texts[0] if answer_texts else ''
                    else:
                        answer_texts = [str(a) for a in answers[:5]]
                        answer = answer_texts[0] if answer_texts else ''
                else:
                    answer = str(answers) if answers else ''
                    answer_texts = [answer] if answer else []
            else:
                answer_texts = [answer]
            
            data_item = {
                "image": image_filename,
                "question": item.get('question', ''),
                "answers": answer_texts[:5] if isinstance(answer_texts, list) else [answer],
                "answer": answer
            }
            
            # Добавляем дополнительную информацию если есть
            if 'question_id' in item:
                data_item['question_id'] = item['question_id']
            elif 'questionId' in item:
                data_item['question_id'] = item['questionId']
            
            data_list.append(data_item)
        
        # Сохраняем JSON
        train_json_path = output_dir / "train.json"
        with open(train_json_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
        
        print(f"✅ VizWiz загружен: {len(data_list)} примеров")
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


def download_docvqa(output_dir="./data/docvqa"):
    """
    Загружает DocVQA датасет
    
    DocVQA можно получить через:
    1. Hugging Face: https://huggingface.co/datasets/ashraq/docvqa
    2. Официальный сайт: https://rrc.cvc.uab.es/?ch=17
    """
    print("=" * 60)
    print("DocVQA Dataset Download")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Вариант 1: Через Hugging Face (рекомендуется)
    try:
        from datasets import load_dataset
        print("📦 Загрузка DocVQA через Hugging Face...")
        
        # Загружаем train split
        dataset = load_dataset("ashraq/docvqa", split="train")
        
        # Сохраняем данные
        data_list = []
        print("📝 Обработка данных...")
        
        for idx, item in enumerate(tqdm(dataset, desc="Processing")):
            # Сохраняем изображение
            image = item['image']
            image_filename = f"docvqa_{idx:06d}.png"
            image_path = images_dir / image_filename
            image.save(image_path)
            
            # Формируем запись в формате, который ожидает наш датасет
            data_item = {
                "image": image_filename,
                "question": item.get('question', ''),
                "answers": item.get('answers', []),
                "answer": item.get('answers', [''])[0] if item.get('answers') else ''
            }
            
            # Также сохраняем оригинальные данные
            data_item.update({
                "questionId": item.get('questionId', ''),
                "ucsf_document_id": item.get('ucsf_document_id', ''),
                "ucsf_document_page_no": item.get('ucsf_document_page_no', 0),
            })
            
            data_list.append(data_item)
        
        # Сохраняем JSON
        train_json_path = output_dir / "train.json"
        with open(train_json_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
        
        print(f"✅ DocVQA загружен: {len(data_list)} примеров")
        print(f"   JSON: {train_json_path}")
        print(f"   Images: {images_dir}")
        return True
        
    except ImportError:
        print("⚠️  Hugging Face datasets не установлен. Установите: pip install datasets")
        print("\nАльтернативный вариант:")
        print("1. Перейдите на https://rrc.cvc.uab.es/?ch=17")
        print("2. Зарегистрируйтесь и загрузите датасет")
        print("3. Распакуйте архив в", output_dir)
        print("4. Убедитесь, что структура следующая:")
        print("   docvqa/")
        print("     train.json")
        print("     images/")
        print("       *.png")
        return False


def download_kvasir_vqa(output_dir="./data/kvasir"):
    print("=" * 60)
    print("Kvasir-VQA Dataset Download")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    print("📦 Kvasir-VQA обычно доступен через GitHub")
    print("   Репозиторий: https://github.com/simula/kvasir-vqa")
    print("\nДля автоматической загрузки попробуем через Hugging Face или прямой URL...")
    
    # Попробуем найти датасет на Hugging Face
    try:
        from datasets import load_dataset
        print("📦 Поиск Kvasir-VQA на Hugging Face...")
        
        # Попробуем различные варианты названий
        dataset_names = [
            "kvasir-vqa",
            "simula/kvasir-vqa",
            "medical-vqa/kvasir",
        ]
        
        dataset = None
        for name in dataset_names:
            try:
                dataset = load_dataset(name, split="train")
                print(f"✅ Найден датасет: {name}")
                break
            except:
                continue
        
        if dataset is None:
            raise ValueError("Датасет не найден на Hugging Face")
        
        # Сохраняем данные
        data_list = []
        print("📝 Обработка данных...")
        
        for idx, item in enumerate(tqdm(dataset, desc="Processing")):
            # Сохраняем изображение
            if 'image' in item:
                image = item['image']
                image_filename = f"kvasir_{idx:06d}.png"
                image_path = images_dir / image_filename
                image.save(image_path)
                
                image_path_str = image_filename
            elif 'image_path' in item:
                image_path_str = item['image_path']
            else:
                continue
            
            # Формируем запись
            data_item = {
                "image": image_path_str,
                "question": item.get('question', item.get('Question', '')),
                "answers": item.get('answers', item.get('Answers', [])),
                "answer": item.get('answer', item.get('Answer', ''))
            }
            
            # Если answers - список, используем первый как answer
            if 'answers' in item and isinstance(item['answers'], list) and len(item['answers']) > 0:
                data_item['answer'] = item['answers'][0]
            elif 'Answers' in item and isinstance(item['Answers'], list) and len(item['Answers']) > 0:
                data_item['answer'] = item['Answers'][0]
            
            data_list.append(data_item)
        
        # Сохраняем JSON
        train_json_path = output_dir / "train.json"
        with open(train_json_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Kvasir-VQA загружен: {len(data_list)} примеров")
        print(f"   JSON: {train_json_path}")
        print(f"   Images: {images_dir}")
        return True
        
    except Exception as e:
        print(f"⚠️  Автоматическая загрузка не удалась: {e}")
        print("\nАльтернативный вариант - ручная загрузка:")
        print("1. Перейдите на https://github.com/simula/kvasir-vqa")
        print("2. Следуйте инструкциям в README")
        print("3. Распакуйте данные в", output_dir)
        print("4. Убедитесь, что структура следующая:")
        print("   kvasir/")
        print("     train.json")
        print("     images/")
        print("       *.jpg или *.png")
        print("\nПример формата train.json:")
        print('  [')
        print('    {')
        print('      "image": "image_001.jpg",')
        print('      "question": "What is visible in the image?",')
        print('      "answers": ["answer1", "answer2"],')
        print('      "answer": "answer1"')
        print('    },')
        print('    ...')
        print('  ]')
        return False


def create_sample_json(output_dir, dataset_name, num_samples=10):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Создаем пример JSON
    sample_data = []
    for i in range(num_samples):
        sample_data.append({
            "image": f"{dataset_name}_sample_{i:03d}.png",
            "question": f"Sample question {i}?",
            "answers": [f"Sample answer {i} A", f"Sample answer {i} B"],
            "answer": f"Sample answer {i} A"
        })
        
        # Создаем пустое изображение для примера
        img_path = images_dir / f"{dataset_name}_sample_{i:03d}.png"
        img = Image.new('RGB', (256, 256), color=(128, 128, 128))
        img.save(img_path)
    
    json_path = output_dir / "train.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Создан пример JSON: {json_path}")
    print(f"   Создано {num_samples} примеров с пустыми изображениями для тестирования")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Загрузка доменных датасетов')
    parser.add_argument('--vqav2', action='store_true', help='Загрузить VQAv2 (рекомендуется)')
    parser.add_argument('--textvqa', action='store_true', help='Загрузить TextVQA (рекомендуется)')
    parser.add_argument('--both', action='store_true', help='Загрузить оба легкодоступных датасета (VQAv2 + TextVQA)')
    parser.add_argument('--clevr', action='store_true', help='Загрузить CLEVR (небольшой синтетический датасет, ~100-200 МБ)')
    parser.add_argument('--vizwiz', action='store_true', help='Загрузить VizWiz (небольшой датасет, ~1-2 ГБ)')
    parser.add_argument('--small-datasets', action='store_true', help='Загрузить оба небольших датасета (CLEVR + VizWiz)')
    parser.add_argument('--docvqa', action='store_true', help='Загрузить DocVQA')
    parser.add_argument('--kvasir', action='store_true', help='Загрузить Kvasir-VQA')
    parser.add_argument('--create-samples', action='store_true', 
                       help='Создать примеры JSON файлов для тестирования')
    parser.add_argument('--output-dir', type=str, default="./data",
                       help='Базовая директория для сохранения (по умолчанию: ./data)')
    
    args = parser.parse_args()
    
    if args.create_samples:
        print("📝 Создание примеров JSON файлов...")
        create_sample_json(f"{args.output_dir}/vqav2", "vqav2", 10)
        create_sample_json(f"{args.output_dir}/textvqa", "textvqa", 10)
        print("\n✅ Примеры созданы. Вы можете использовать их для тестирования.")
        print("   После загрузки реальных данных, замените train.json на реальные данные.")
        return
    
    success_count = 0
    if args.both or args.vqav2:
        if download_vqav2(f"{args.output_dir}/vqav2"):
            success_count += 1
    
    if args.both or args.textvqa:
        if download_textvqa(f"{args.output_dir}/textvqa"):
            success_count += 1
    
    if args.small_datasets or args.clevr:
        if download_clevr(f"{args.output_dir}/clevr"):
            success_count += 1
    
    if args.small_datasets or args.vizwiz:
        if download_vizwiz(f"{args.output_dir}/vizwiz"):
            success_count += 1
    
    if args.docvqa:
        if download_docvqa(f"{args.output_dir}/docvqa"):
            success_count += 1
    
    if args.kvasir:
        if download_kvasir_vqa(f"{args.output_dir}/kvasir"):
            success_count += 1
    
    if success_count == 0:
        print("\n⚠️  Не удалось автоматически загрузить датасеты.")
        print("Используйте --create-samples для создания примеров JSON файлов.")
        print("Или загрузите датасеты вручную согласно инструкциям выше.")


if __name__ == "__main__":
    main()

