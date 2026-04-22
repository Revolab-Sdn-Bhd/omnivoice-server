# Voice Samples

Reference audio files for voice cloning profiles. Each `.wav` has a companion `.txt`
containing the exact transcript required when registering the profile.

## Samples

| File | Language | ref_text |
|------|----------|----------|
| `anwar.wav` | Malay | see `anwar.txt` |

## Adding a profile from these samples

```bash
curl -X POST http://localhost:8880/v1/voices/profiles \
  -F "profile_id=anwar" \
  -F "ref_audio=@voice_samples/anwar.wav" \
  -F "ref_text=$(cat voice_samples/anwar.txt)"
```

> `ref_text` must be the **exact** transcript of the reference audio.
> It conditions the speaker embedding — wrong text degrades clone quality.
