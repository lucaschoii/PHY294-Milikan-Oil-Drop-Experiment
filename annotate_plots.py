import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
data_path = '/Users/lucaschoi/Documents/GitHub/PHY294-Milikan-Oil-Drop-Experiment/data'
files = glob.glob(os.path.join(data_path, '*.tsv'))

output_path = '/Users/lucaschoi/Documents/GitHub/PHY294-Milikan-Oil-Drop-Experiment/annotations.tsv'
annotation_df = pd.read_csv(output_path, sep='\t', header=None, names=['filename', 'rise_start', 'rise_end', 'fall_start', 'fall_end'])
file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0


# Store annotations with frame numbers for each phase
annotations = {
    # Format: {filename: {'rise_start': x1, 'rise_end': x2, 'fall_start': x3, 'fall_end': x4}}
}

def plot_file(file):
    """Plots a TSV file and allows annotation of rising/falling periods."""
    # Load data
    df = pd.read_csv(file, sep='\t', header=None, names=['y-position'], skiprows=1)
    x = [i for i in range(len(df))]  # Time in seconds
    y = [val  for val in pd.to_numeric(df['y-position'], errors='coerce')]  # Converts to NumPy array, then multiplies  # Converts '#NV' to NaN, y-position in m
    filename = os.path.basename(file)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, label=filename)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Y-Position (m)")
    ax.set_title(f"Click to annotate {filename}")

    # Initialize annotation state
    click_count = 0
    phases = ['rise_start', 'rise_end', 'fall_start', 'fall_end']
    current_phase = None
    vlines = []  # Store reference to vertical lines
    
    def onclick(event):
        global annotation_df
        nonlocal click_count, current_phase, vlines
        if event.inaxes != ax:
            return
        
        clicked_frame = int(event.xdata)
        
        # Clear previous vertical lines
        for vline in vlines:
            vline.remove()
        vlines = []
        
        if click_count < 4:
            current_phase = phases[click_count]
            annotations.setdefault(filename, {})[current_phase] = clicked_frame  # Store frame
            click_count += 1
            
            # Visual feedback (draw at clicked time)
            color = 'green' if 'rise' in current_phase else 'red'
            vlines.append(ax.axvline(clicked_frame, color=color, linestyle='--', alpha=0.7))
            
            # Update title to show next expected click
            if click_count < 4:
                next_phase = phases[click_count].replace('_', ' ')
                ax.set_title(f"Next: {next_phase}\nFile: {filename}")
            else:
                print(f"All phases annotated for {filename}.")
                # Save annotations to annotations_df
                new_row = {
                    'filename': filename,
                    'rise_start': annotations[filename]['rise_start'],
                    'rise_end': annotations[filename]['rise_end'],
                    'fall_start': annotations[filename]['fall_start'],
                    'fall_end': annotations[filename]['fall_end'],
                }
                current_annotation = pd.DataFrame([new_row])
                # Save to CSV
                current_annotation.to_csv(output_path, sep='\t', index=False, mode='a', header = not file_exists)
                
            
            plt.draw()
        else:
            plt.close(fig)

    
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


# Process files
print(f"Found {len(files)} data files")
skipped = 0
annotated = 0
for i, file in enumerate(files, 1):
    # Skip already annotated files
    filename = os.path.basename(file)
    if filename in annotation_df['filename'].values:
        print(f"Skipping {filename} (already annotated)")
        skipped += 1
        continue

    else:
        print(f"File {filename} not found in annotations. Proceeding to annotate.")
        plot_file(file)
        annotated += 1

print(f"\nAnnotated {annotated} files, skipped {skipped} files.")
print("All files processed.")
        
