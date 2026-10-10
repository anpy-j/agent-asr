import array
import math
import wave

import av
import pytest

from agent_asr.core import prepare_audio


def test_resample_stereo_long_audio_without_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "")
    source = tmp_path / "立体声音频.wav"
    output = tmp_path / "mono.wav"
    rate, seconds = 44100, 61
    block = array.array("h")
    for i in range(rate):
        value = round(8000 * math.sin(2 * math.pi * 440 * i / rate))
        block.extend([value, value])
    with wave.open(str(source), "wb") as handle:
        handle.setparams((2, 2, rate, 0, "NONE", "not compressed"))
        for _ in range(seconds):
            handle.writeframesraw(block.tobytes())
    assert prepare_audio(source, output) == seconds
    with wave.open(str(output), "rb") as handle:
        assert (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) == (1, 2, 16000)
        assert handle.getnframes() == seconds * 16000
        for second in (0, 30, 60):
            handle.setpos(second * 16000 + 1600)
            pcm = array.array("h", handle.readframes(1600))
            rms = math.sqrt(sum(v * v for v in pcm) / len(pcm))
            assert 5000 < rms < 9000  # signal survives downmixing throughout the file


def test_compressed_audio_in_mp4_flushes_tail(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "")
    source, output = tmp_path / "input.mp4", tmp_path / "output.wav"
    with av.open(str(source), "w") as container:
        stream = container.add_stream("aac", rate=48000)
        stream.layout = "mono"
        for _ in range(100):
            frame = av.AudioFrame(format="s16", layout="mono", samples=960)
            frame.sample_rate = 48000
            frame.planes[0].update(array.array("h", [4000] * 960).tobytes())
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    duration = prepare_audio(source, output)
    assert duration == pytest.approx(2, abs=0.05)  # AAC padding is container-dependent
    with wave.open(str(output), "rb") as handle:
        handle.setpos(int(1.8 * 16000))
        tail = array.array("h", handle.readframes(1600))
        assert max(tail) > 1000


def test_no_audio_and_corrupt_input_remove_partial_output(tmp_path):
    source, output = tmp_path / "empty.wav", tmp_path / "output.wav"
    with wave.open(str(source), "wb") as handle:
        handle.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
    with pytest.raises(RuntimeError, match="Audio decoding failed"):
        prepare_audio(source, output)
    assert not output.exists()
    source.write_bytes(b"not media")
    with pytest.raises(RuntimeError, match="Audio decoding failed"):
        prepare_audio(source, output)
    assert not output.exists()
    source = tmp_path / "video.mp4"
    with av.open(str(source), "w") as container:
        stream = container.add_stream("mpeg4", rate=25)
        stream.width, stream.height = 16, 16
        for packet in stream.encode(av.VideoFrame(16, 16, "yuv420p")):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    with pytest.raises(RuntimeError, match="no audio stream"):
        prepare_audio(source, output)
    assert not output.exists()
