import os
import argparse
import warnings

import torch
import torchaudio

from voicedit import VoiceDiTPipeline

warnings.filterwarnings("ignore")   # ignore warning


def parse_args(parser):
    parser.add_argument("--modality", type=str, default="text", help="modality for generation. 'text' or 'audio' or 'image'")
    parser.add_argument('--desc_prompt', '-d', type=str, help='description prompt')
    parser.add_argument('--cont_prompt', '-c', type=str, help='content prompt')
    parser.add_argument('--speaker_audio', '-spk', type=str, help='speaker audio')

    parser.add_argument("--ckpt_path", type=str, required=True, help="checkpoint file path for VoiceDiT")
    parser.add_argument("--v2a_ckpt_path", type=str, help='checkpoint file path for V2A-Mapper')
    parser.add_argument("--output_dir", type=str, default="./outputs", help="directory to save generated audio")
    parser.add_argument("--file_name", type=str, help="filename for the generated audio")

    parser.add_argument('--num_inference_steps', type=int, default=100, help='number of inference steps for DDIM sampling')
    parser.add_argument('--audio_length_in_s', type=float, default=10, help='duration of the audio for generation')
    parser.add_argument('--v2a_guidance_scale', type=float, default=2.0, help='guidance weight for v2a-mapper classifier-free guidance')
    parser.add_argument('--guidance_scale', type=float, help='guidance weight for single classifier-free guidance')
    parser.add_argument('--desc_guidance_scale', type=float, default=5, required=False, help='desc guidance weight for dual classifier-free guidance')
    parser.add_argument('--cont_guidance_scale', type=float, default=5, required=False, help='cont guidance weight for dual classifier-free guidance')
    parser.add_argument('--female_voice_list', type=str, default='libri_test_clean_female.txt')
    parser.add_argument('--male_voice_list', type=str, default='libri_test_clean_male.txt')
    
    parser.add_argument("--device", type=str, default="auto", help="device to use for audio generation")
    parser.add_argument('--seed', type=int, help='random seed for generation')

    return parser.parse_args()


def main():
    parser = argparse.ArgumentParser()
    args = parse_args(parser)
    
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = torch.device("cuda:0")
        else:
            args.device = torch.device("cpu")
    elif args.device is not None:
        args.device = torch.device(args.device)
    else:
        args.device = torch.device("cpu")

    pipe = VoiceDiTPipeline(
        args.ckpt_path,
        t2a_ckpt_path = args.t2a_ckpt_path,
        v2a_ckpt_path = args.v2a_ckpt_path,
        device = args.device,
        male_voice = args.male_voice_list,
        female_voice = args.female_voice_list,
    )
    
    audio, _ = pipe(
        modality = args.modality,
        env_prompt = args.desc_prompt,
        cont_prompt = args.cont_prompt,
        batch_size = 1,
        num_inference_steps = args.num_inference_steps,
        audio_length_in_s = 10,
        do_classifier_free_guidance = True,
        desc_guidance_scale = args.desc_guidance_scale,
        cont_guidance_scale = args.cont_guidance_scale,
        v2a_guidance_scale = args.v2a_guidance_scale,
        device=args.device,
        seed=args.seed,
        progress=True,
        speaker_audio=args.speaker_audio,
    )
    
    file_name = args.file_name
    if file_name is None:
        if args.modality == "text":
            file_name = args.desc_prompt[:10]
        else:
            file_name = os.path.basename(args.desc_prompt[:-4])
        file_name = file_name + "-" + args.cont_prompt[:10] + ".wav"

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, file_name)
        
    torchaudio.save(save_path, src=audio, sample_rate=16000)


if __name__ == "__main__":
    main()