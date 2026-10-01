use anyhow::{Result, ensure};
use rustfft::{FftPlanner, num_complex::Complex};
use std::path::Path;

pub const SAMPLES: usize = 480_000;
pub const FRAMES: usize = 3000;
pub const MELS: usize = 80;

pub fn read_wav(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let s = reader.spec();
    ensure!(
        s.sample_rate == 16000,
        "WAV must be 16 kHz (got {})",
        s.sample_rate
    );
    ensure!(
        s.channels == 1 || s.channels == 2,
        "WAV must be mono or stereo"
    );
    ensure!(
        reader.duration() as usize <= SAMPLES,
        "WAV exceeds 30 seconds; split it before transcription"
    );
    let values = match s.sample_format {
        hound::SampleFormat::Float => {
            ensure!(
                s.bits_per_sample == 32,
                "only 32-bit float WAV is supported"
            );
            reader
                .samples::<f32>()
                .collect::<std::result::Result<Vec<_>, _>>()?
        }
        hound::SampleFormat::Int => {
            ensure!(
                [8, 16, 24, 32].contains(&s.bits_per_sample),
                "unsupported PCM bit depth"
            );
            let scale = 2f32.powi(s.bits_per_sample as i32 - 1);
            reader
                .samples::<i32>()
                .map(|v| v.map(|x| x as f32 / scale))
                .collect::<std::result::Result<Vec<_>, _>>()?
        }
    };
    ensure!(
        !values.is_empty() && values.len().is_multiple_of(s.channels as usize),
        "empty or incomplete WAV frames"
    );
    ensure!(
        values.iter().all(|x| x.is_finite()),
        "WAV contains nonfinite samples"
    );
    let mono: Vec<_> = values
        .chunks_exact(s.channels as usize)
        .map(|v| v.iter().map(|x| x / s.channels as f32).sum::<f32>())
        .collect();
    ensure!(
        mono.len() <= SAMPLES,
        "WAV exceeds 30 seconds; split it before transcription"
    );
    Ok(mono)
}

fn hz_to_mel(hz: f64) -> f64 {
    if hz < 1000. {
        hz / (200. / 3.)
    } else {
        15. + (hz / 1000.).ln() / (6.4f64.ln() / 27.)
    }
}
fn mel_to_hz(mel: f64) -> f64 {
    if mel < 15. {
        mel * (200. / 3.)
    } else {
        1000. * ((mel - 15.) * (6.4f64.ln() / 27.)).exp()
    }
}
/// Time-major [3000,80], for the Hex frontend. Matches Whisper's centered STFT.
pub fn log_mel(samples: &[f32]) -> Result<Vec<f32>> {
    ensure!(
        !samples.is_empty() && samples.len() <= SAMPLES && samples.iter().all(|x| x.is_finite()),
        "invalid audio samples"
    );
    let mut padded = vec![0f32; SAMPLES];
    padded[..samples.len()].copy_from_slice(samples);
    let edges: Vec<_> = (0..MELS + 2)
        .map(|i| mel_to_hz(hz_to_mel(8000.) * i as f64 / (MELS + 1) as f64))
        .collect();
    let mut filters = vec![vec![0f32; 201]; MELS];
    for m in 0..MELS {
        for (k, filter) in filters[m].iter_mut().enumerate() {
            let hz = k as f64 * 40.;
            *filter = (((hz - edges[m]) / (edges[m + 1] - edges[m]))
                .min((edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1]))
                .max(0.)
                * 2.
                / (edges[m + 2] - edges[m])) as f32;
        }
    }
    let fft = FftPlanner::new().plan_fft_forward(400);
    let window: Vec<_> = (0..400)
        .map(|i| (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / 400.).cos()) as f32)
        .collect();
    let mut buf = vec![Complex::default(); 400];
    let mut scratch = vec![Complex::default(); fft.get_inplace_scratch_len()];
    let mut output = vec![0.; FRAMES * MELS];
    for frame in 0..FRAMES {
        for i in 0..400 {
            let index = frame as isize * 160 + i as isize - 200;
            let index = if index < 0 {
                -index
            } else if index >= SAMPLES as isize {
                2 * SAMPLES as isize - 2 - index
            } else {
                index
            } as usize;
            buf[i] = Complex::new(padded[index] * window[i], 0.);
        }
        fft.process_with_scratch(&mut buf, &mut scratch);
        let power: Vec<_> = buf[..201].iter().map(|v| v.norm_sqr()).collect();
        for m in 0..MELS {
            output[frame * MELS + m] = filters[m]
                .iter()
                .zip(&power)
                .map(|(f, p)| f * p)
                .sum::<f32>()
                .max(1e-10)
                .log10();
        }
    }
    ensure!(
        output.iter().all(|v| v.is_finite()),
        "audio amplitude overflowed feature extraction"
    );
    let floor = output.iter().copied().fold(f32::NEG_INFINITY, f32::max) - 8.;
    output
        .iter_mut()
        .for_each(|v| *v = (v.max(floor) + 4.) / 4.);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn silence_and_invalid_samples() {
        let mel = log_mel(&[0.; 160]).unwrap();
        assert_eq!(mel.len(), FRAMES * MELS);
        assert!(mel.iter().all(|v| *v == -1.5));
        assert!(log_mel(&[]).is_err());
        assert!(log_mel(&[f32::NAN]).is_err());
        assert!(log_mel(&vec![0.; SAMPLES + 1]).is_err());
    }
}
