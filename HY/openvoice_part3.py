import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS
import nltk
nltk.download('averaged_perceptron_tagger_eng')


# 경로 설정
ckpt_converter = r'C:\Users\arh05\openvoice\checkpoints_v2_0417\checkpoints_v2\converter'
base_speakers_dir = r'C:\Users\arh05\openvoice\checkpoints_v2_0417\checkpoints_v2\base_speakers\ses'
output_dir = r'C:\Users\arh05\openvoice\outputs'

# Device 설정
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ToneColorConverter 초기화
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

# Reference Speaker 설정
reference_speaker = r'C:\Users\arh05\openvoice\OpenVoice\resources\demo_speaker2.wav'

# Target SE 추출
print(f"Extracting SE from: {reference_speaker}")
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=False)

# TTS 생성 및 톤 컬러 변환
texts = {
    'EN_NEWEST': "Did you ever hear a folk tale about a giant turtle?",
    'EN': "Did you ever hear a folk tale about a giant turtle?",
    'ES': "El resplandor del sol acaricia las olas, pintando el cielo con una paleta deslumbrante.",
    'FR': "La lueur dorée du soleil caresse les vagues, peignant le ciel d'une palette éblouissante.",
    'ZH': "在这次vacation中，我们计划去Paris欣赏埃菲尔铁塔和卢浮宫的美景。",
    'JP': "彼は毎朝ジョギングをして体を健康に保っています。",
    'KR': "안녕하세요! 저는 일론머스크입니다. 저는 한국의 파기차차를 존경하며, 그에게서 많은 영감을 받았습니다. 저는 그를 멘토 삼아 화성을 제 2의 지구로 만들것입니다. 감사합니다.",
}

src_path = os.path.join(output_dir, 'tmp.wav')
speed = 1.0

print("Processing TTS and tone conversion...")
for language, text in texts.items():
    try:
        model = TTS(language=language, device=device)
        speaker_ids = model.hps.data.spk2id

        for speaker_key in speaker_ids.keys():
            speaker_id = speaker_ids[speaker_key]
            speaker_key = speaker_key.lower().replace('_', '-')

            # SE 파일 경로
            source_se_path = os.path.join(base_speakers_dir, f"{speaker_key}.pth")

            if not os.path.exists(source_se_path):
                print(f"Skipping {speaker_key}, SE file not found: {source_se_path}")
                continue

            # SE 파일 로드
            source_se = torch.load(source_se_path, map_location=device)

            # TTS 변환
            model.tts_to_file(text, speaker_id, src_path, speed=speed)

            # Tone 컬러 변환
            save_path = os.path.join(output_dir, f'output_v2_{speaker_key}.wav')
            encode_message = "@MyShell"
            tone_color_converter.convert(
                audio_src_path=src_path,
                src_se=source_se,
                tgt_se=target_se,
                output_path=save_path,
                message=encode_message
            )
            print(f"Processed: {save_path}")
    except Exception as e:
        print(f"Error processing language {language}: {e}")

print("Processing complete.")
