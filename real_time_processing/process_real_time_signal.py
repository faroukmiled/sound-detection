import argparse
import queue
import sys

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import librosa,os

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "-l",
    "--list-devices",
    action="store_true",
    help="show list of audio devices and exit",
)
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser],
)
parser.add_argument(
    "channels",
    type=int,
    default=[1],
    nargs="*",
    metavar="CHANNEL",
    help="input channels to plot (default: the first)",
)
parser.add_argument(
    "-d", "--device", type=int_or_str, help="input device (numeric ID or substring)"
)
parser.add_argument(
    "-w",
    "--window",
    type=float,
    default=100,
    metavar="DURATION",
    help="visible time slot (default: %(default)s ms)",
)
parser.add_argument(
    "-i",
    "--interval",
    type=float,
    default=30,
    help="minimum time between plot updates (default: %(default)s ms)",
)
parser.add_argument("-b", "--blocksize", type=int, help="block size (in samples)")
parser.add_argument(
    "-r",
    "--samplerate",
    type=float,
    help="sampling rate of audio device"
)
parser.add_argument(
    "-n",
    "--downsample",
    type=int,
    default=10,
    metavar="N",
    help="display every Nth sample (default: %(default)s)",
)
args = parser.parse_args(remaining)
if any(c < 1 for c in args.channels):
    parser.error("argument CHANNEL: must be >= 1")
mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1
q = queue.Queue()

# Additional code to process the signal in real time
def process_signal(indata):
    processed_data=indata
    time_values=[]
    # Your processing code here
    #print("Processing signal:", indata)
    target_sound_file = "./tt_samples/ping1.wav"
    # Load the target sound effect
    target_sound,sr = librosa.load(target_sound_file)
    target_sound = np.array(target_sound)
    # Load the audio or video file
    
    length_video=len(indata)
    print(length_video)

    segment_duration = int(len(target_sound)*1000/ sr)
    print(segment_duration)
    # Step 2: Convert the target sound effect to a spectrogram# Step 3: Split the audio into short segments and compare them with the target sound effect  # Convert milliseconds to seconds)
    segment_length = segment_duration
    l=[]
    samples=os.listdir("./tt_samples")
    for i in range(0, length_video - segment_length,10):
        l1=[]
        # Extract a segment from the audio
        segment = indata[i:i + segment_length]

        for j in range(len(samples)):
            sample=samples[j]
            audio_file=os.path.join('./tt_samples',sample)
            target_sound,sr = librosa.load(audio_file)
            target_sound = np.array(target_sound)
            target_sound_spec = librosa.amplitude_to_db(np.abs(librosa.stft(target_sound, hop_length=512)), ref=np.max)

        # Convert the segment to a spectrogram using librosa
            segment_spec = librosa.amplitude_to_db(np.abs(librosa.stft(segment, hop_length=512)), ref=np.max)

        # Compare the spectrograms of the segment and target sound effect
            resized_target_sound_spec = np.resize(target_sound_spec, segment_spec.shape)

        # Compare the spectrograms of the segment and target sound effect
            similarity = np.mean(np.abs(segment_spec - resized_target_sound_spec))

        # Set a threshold to determine if the target sound effect is present
            threshold = 20 # Adjust this value based on your requirements

            if similarity < threshold:
                l1.append((similarity,j))
        if l1!=[]:
             time_values.append(i)
    return processed_data,time_values
            

def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy:
    processed_data, time_values = process_signal(indata)
    q.put((processed_data[:: args.downsample, mapping], time_values))




def update_plot(frame):
    """This is called by matplotlib for each plot update.
    
    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.

    """
    global plotdata
    while True:
        try:
            data,time_values= q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data
        for time in range(len(data)):
        # Check if current time value is in the list of time values
            if time in time_values:
                print("YES")
                ax.set_facecolor('green')  # Highlight the plot in green
            else:
                ax.set_facecolor('white')  # Set plot background to white
        
    for column, line in enumerate(lines):
        line.set_ydata(plotdata[:, column])
    return lines


try:
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, "input")
        args.samplerate = device_info["default_samplerate"]

    length = int(args.window * args.samplerate / (1000 * args.downsample))
    plotdata = np.zeros((length, len(args.channels)))

    fig, ax = plt.subplots()
    lines = ax.plot(plotdata)
    if len(args.channels) > 1:
        ax.legend(
            [f"channel {c}" for c in args.channels],
            loc="lower left",
            ncol=len(args.channels),
        )
    ax.axis((0, len(plotdata), -1, 1))
    ax.set_yticks([0])
    ax.yaxis.grid(True)
    ax.tick_params(
        bottom=False,
        top=False,
        labelbottom=False,
        right=False,
        left=False,
        labelleft=False,
    )
    fig.tight_layout(pad=0)

    stream = sd.InputStream(
        device=args.device,
        channels=max(args.channels),
        samplerate=args.samplerate,
        callback=audio_callback,
    )
    time_values=[100,200,300,400]
    ani = FuncAnimation(fig, update_plot, interval=args.interval, blit=True)
    with stream:
        plt.show()
except Exception as e:
    parser.exit(type(e).__name__ + ": " + str(e))
