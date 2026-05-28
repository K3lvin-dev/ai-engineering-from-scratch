# Clonagem e Conversão de Voz

> Clonagem de voz lê seu texto na voz de outra pessoa. Conversão de voz reescreve sua voz na de outra pessoa preservando o que você disse. Ambos dependem da mesma decomposição: separar identidade do falante do conteúdo.

**Tipo:** Construir
**Idiomas:** Python
**Pré-requisitos:** Fase 6 · 06 (Reconhecimento de Falante), Fase 6 · 07 (TTS)
**Tempo:** ~75 minutos

## O Problema

Em 2026, um clipe de áudio de 5 segundos é suficiente para produzir uma clonagem de alta qualidade da voz de qualquer pessoa com uma GPU de consumidor. ElevenLabs, F5-TTS, OpenVoice v2, VoiceBox todos oferecem clonagem zero-shot ou few-shot. A tecnologia é bênção (TTS acessível, dublagem, vozes assistivas) e arma (ligações de golpe, deepfakes políticos, roubo de IP).

Duas tarefas intimamente relacionadas:

- **Clonagem de voz (lado TTS):** texto + voz de referência de 5 segundos → áudio nessa voz.
- **Conversão de voz (lado fala):** áudio fonte (pessoa A dizendo X) + voz de referência da pessoa B → áudio de B dizendo X.

Ambos fatoram uma forma de onda em (conteúdo, falante, prosódia) e recombinam conteúdo de uma fonte com falante de outra.

Restrição-chave sob a qual você opera em 2026: **marcas d'água e gates de consentimento são obrigatórios legalmente na UE (AI Act, aplicável agosto de 2026) e na Califórnia (AB 2905, vigente em 2025)**. Sua pipeline deve emitir uma marca d'água inaudível e recusar clones não-consensuais.

## O Conceito

![Clonagem vs conversão de voz: fatorizar, trocar falante, recombinar](../assets/voice-cloning.svg)

**Clonagem zero-shot.** Passe um clipe de 5 segundos para um modelo treinado em milhares de falantes. O encoder de falante mapeia o clipe para um embedding de falante; o decoder TTS condiciona nesse embedding mais texto.

Usado por: F5-TTS (2024), YourTTS (2022), XTTS v2 (2024), OpenVoice v2 (2024).

**Ajuste fino few-shot.** Grave 5-30 minutos da voz alvo. Ajuste fino LoRA em um modelo base por uma hora. Qualidade salta de "ok" a "indistinguível". Coqui e ElevenLabs suportam esse padrão; comunidade usa com F5-TTS.

**Conversão de voz (VC).** Duas famílias:

- **Reconhecimento-síntese.** Rode modelo parecido com ASR para extrair representação de conteúdo (ex. posteriores fonêmicos moles, PPGs), depois resintetize com embedding do falante alvo. Robusto a idioma e sotaque. Usado por KNN-VC (2023), Diff-HierVC (2023).
- **Desentrelaçamento.** Treine um autoencoder que separa conteúdo, falante e prosódia no espaço latente no gargalo. Troque embedding do falante na inferência. Menor qualidade mas mais rápido. Usado por AutoVC (2019), variantes VITS-VC.

**Clonagem baseada em codec neural (2024+).** VALL-E, VALL-E 2, NaturalSpeech 3, VoiceBox — tratam áudio como tokens discretos de SoundStream / EnCodec, treinam um modelo autoregressivo ou flow-matching grande sobre tokens de codec. Qualidade comparável ao ElevenLabs em prompts curtos.

### A parte ética, não é acessório

**Marcas d'água.** PerTh (Perth) e SilentCipher (2024) embutem um ID de ~16-32 bits imperceptivelmente no áudio. Sobrevive a re-encoding, streaming e edições comuns. Open source pronto para produção.

**Gates de consentimento.** Precisa parear cada saída clonada com um registro de consentimento verificável. Armazene em log à prova de violação.

**Detecção.** AASIST, RawNet2 e Wav2Vec2-AASIST são disponibilizados como detectores. O desafio ASVspoof 2025 publicou EERs de 0,8–2,3% para detectores SOTA contra ElevenLabs, VALL-E 2 e saídas do Bark.

### Números (2026)

| Modelo | Zero-shot? | SECS (similaridade alvo) | WER (inteligibilidade) | Params |
|--------|-----------|--------------------------|----------------------|--------|
| F5-TTS | Sim | 0,72 | 2,1% | 335M |
| XTTS v2 | Sim | 0,65 | 3,5% | 470M |
| OpenVoice v2 | Sim | 0,70 | 2,8% | 220M |
| VALL-E 2 | Sim | 0,77 | 2,4% | 370M |
| VoiceBox | Sim | 0,78 | 2,1% | 330M |

SECS > 0,70 é geralmente indistinguível do alvo para a maioria dos ouvintes.

## Construa

### Passo 1: decomponha com reconhecimento-síntese

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

Conceitualmente simples; a massa de implementação está em `tts_model` e no encoder de falante.

### Passo 2: clonagem zero-shot com F5-TTS

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

A transcrição de referência deve coincidir exatamente com o áudio; mismatch quebra o alinhamento.

### Passo 3: conversão de voz com KNN-VC

```python
import torch
from knnvc import KNNVC
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC roda WavLM para extrair embeddings por frame para fonte e pool alvo, depois substitui cada frame fonte por seu vizinho mais próximo no pool. Não-paramétrico, funciona com um minuto de fala alvo.

### Passo 4: embuta uma marca d'água

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)
```

~32 bits de payload, detectável após re-encode MP3 e ruído leve.

### Passo 5: gate de consentimento

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## Use

A pilha de 2026:

| Situação | Escolha |
|----------|---------|
| Clonagem zero-shot de 5 s, open-source | F5-TTS ou OpenVoice v2 |
| Clonagem produção comercial | ElevenLabs Instant Voice Clone v2.5 |
| Conversão de voz (reescrever) | KNN-VC ou Diff-HierVC |
| Ajuste fino multi-falante | StyleTTS 2 + adaptador de falante |
| Clonagem cross-lingual | XTTS v2 ou VALL-E X |
| Detecção de deepfake | Wav2Vec2-AASIST |

## Armadilhas

- **Transcrição de referência desalinhada.** F5-TTS e similares exigem que o texto de referência combine exatamente com o áudio, pontuação incluída.
- **Referência com reverberação.** Eco mata a clonagem. Grave seco, microfone próximo.
- **Incompatibilidade emocional.** Referência de treino "alegre" produz clones alegres de tudo. Combine emoção da referência com uso alvo.
- **Vazamento de idioma.** Clonar um falante de inglês e depois pedir para falar francês geralmente carrega o sotaque; use modelos cross-lingual (XTTS, VALL-E X).
- **Sem marca d'água.** Ilegalmente inviável na UE a partir de agosto de 2026.

## Entregue

Salve como `outputs/skill-voice-cloner.md`. Projete uma pipeline de clonagem ou conversão com gate de consentimento + marca d'água + objetivo de qualidade.

## Exercícios

1. **Fácil.** Execute `code/main.py`. Demonstra a troca de embedding de falante computando o cosseno entre dois "falantes" antes e depois da troca.
2. **Médio.** Use OpenVoice v2 para clonar sua própria voz. Meça SECS entre referência e clone. Meça CER via Whisper.
3. **Difícil.** Aplique marca d'água SilentCipher em 20 clones, rode encode+decode MP3 a 128 kbps, detecte o payload. Reporte acurácia em bits.

## Termos Chave

| Termo | O que a gente diz | O que significa de verdade |
|-------|-------------------|---------------------------|
| Clonagem zero-shot | 5 segundos bastam | Modelo pré-treinado + embedding de falante; sem treino. |
| PPG | Posteriori fonêmico | Posteriores ASR por frame usados como representação de conteúdo agnóstica a idioma. |
| KNN-VC | Conversão por vizinho mais próximo | Substitui cada frame fonte pelo frame mais próximo do pool alvo. |
| TTS de codec neural | Estilo VALL-E | Modelo AR sobre tokens EnCodec/SoundStream. |
| Marca d'água | Assinatura inaudível | Bits embutidos no áudio, sobrevivem a re-encode. |
| SECS | Fidelidade de clonagem | Cosseno entre embeddings do falante alvo e clone. |
| AASIST | Detector de deepfake | Modelo anti-spoof; detecta fala sintetizada. |

## Leitura Adicional

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — clonagem zero-shot SOTA open-source.
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) e [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) — TTS de codec neural.
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) — conversão de voz baseada em desentrelaçamento.
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) — VC baseada em recuperação.
- [SilentCipher (2024) — Audio Watermarking](https://github.com/sony/silentcipher) — marca d'água de áudio de 32 bits pronta para produção.
- [Resultados ASVspoof 2025](https://www.asvspoof.org/) — corrida armamentista detector vs sintetizador, atualizado 2026.
