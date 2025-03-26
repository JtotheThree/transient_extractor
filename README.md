Python Script to extract all detected transients from a sample or folder of samples and save them into sample chains.

Inspired by past drum layering and re-insprired by the transient feature on the Elektron Digitone II.

I'm using this to pull transients into sample chains and then uploading them to a Machinedrum.

For Machinedrum/Elektron usage upload to ROM 24-48 for linear start points and load it into a ROM machine with very low Decay and Hold set. Play with the Start parameter.

The --sort flag will break up the transients into low (<1000hz), med (> 1000hz and < 4000hz) and high (>4000hz). This will give a decent range I've found to look for transients to layer over kick, snare, hats etc.

Use the --randomize to shuffle up the ordering of the transients. Useful if you're extracting a lot of samples and don't want all the similar sounds next to each other.

Transient detection is done with [Librosa](https://github.com/librosa/librosa). Adjust the script calls for in the script if you want to alter how it does detection. I left it as default since I couldn't find any settings that handle the task better than the defaults.

### How To Use
Follow these steps to clone the repository and set up the environment to run the script:

1. **Clone the Repository**  
   First, clone the repository to your local machine using Git:

   ```bash
   git clone https://github.com/JtotheThree/transient_extractor.git
   cd transient_extractor
   # Create and activate a python virtual environment if desired
   pip install -r requirments.txt
   ```

# Example
```bash
python transient_extractor.py --input "MyLargeSampleFolder/" --recurse -output "Transients" -length 512 --sort --randomize --gap
```

Example for Machinedrum
```bash
python transient_extractor.py --input "DrumBreak.wav" --output "Transients" --length 256 --syx
```

# Requires
You'll need [SoX]https://github.com/chirlu/sox] installed to use the --syx flag to convert files to .syx files for midi transfer. 
Install with your Linux or Mac package manager or [SoX Download](https://sourceforge.net/projects/sox/) for Windows.

### Command Line Arguments

```bash
usage: transient_extractor.py [-h] [-i INPUT] [-r] [-o OUTPUT]
                                     [-l LENGTH] [--gap] [--sort]
                                     [--randomize] [--syx] [--plot]
```
Extract transient features from given audio file(s).

    -i INPUT, --input: Path to the input audio file(s). Can be a single file or a directory containing .wav files.

    -r, --recurse: Recursively load audio files in the input directory. Default is false (does not recurse).

    -o OUTPUT, --output: Path to the output folder where the extracted transients will be saved.

    -l LENGTH, --length: Length of the transient in samples. Default is 512.

    --gap: Adds a gap between transients the same size as the transient.

    --sort: Sort the transients by spectral centroid into low, mid, and high frequency ranges (kick, snare, hat).

    --randomize: Randomizes the order of the transients.

    --syx: Output transients as SDS-compatible SysEx files.

    --plot: Plot the detected transients over the waveform of the audio file. Using this with multiple files will plot EACH file consecutively.