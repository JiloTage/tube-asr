# src/decoder.py
"""
このモジュールは、wav2vec2モデル向けのShallow Fusion実装例を示します。
pyctcdecodeを利用し、KenLMによる外部LMスコアを統合します。
Whisperの場合は直接的なShallow Fusion実装が難しいため、ここではwav2vec2のみサポート例とします。
"""

from pyctcdecode import build_ctcdecoder

def build_decoder(processor, lm_path, lm_weight, beam_width):
    """
    processor: Wav2Vec2Processor（トークナイザ含む）
    lm_path: kenlm形式のLMファイルパス
    lm_weight: LMの重み λ
    beam_width: ビームサーチ幅
    """
    vocab = list(processor.tokenizer.get_vocab().keys())
    # pyctcdecodeのdecoder生成。ここでは簡易にスペシャルトークン等は除去した前提
    decoder = build_ctcdecoder(
        labels=vocab,
        kenlm_model_path=lm_path,
        alpha=lm_weight,    # LM重み
        beta=0.0,           # 調整パラメータ（今回は0）
        beam_width=beam_width
    )
    return decoder

def decode_logits(logits, decoder):
    """
    logits: モデル出力のlogits (numpy配列想定)
    decoder: build_decoderで生成したデコーダ
    """
    # CTCデコーディングを実行
    transcription = decoder.decode(logits)
    return transcription
