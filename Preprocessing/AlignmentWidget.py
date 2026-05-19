import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, IntSlider, Layout
import ipywidgets as widgets
from scipy.ndimage import shift, rotate
import hyperspy.api as hs
import matplotlib.colors as mcolors
import os 

# plot images in widget + blended overlay
class ImageAligner:
    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2
        self.aligned = img2
        self.shift_x = 0
        self.shift_y = 0
        self.rotation_angle = 0
      
    def plot_images(self, shift_x=0, shift_y=0,alpha=0.5, rotation=0):
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.rotation_angle = rotation
        #plt.close('all') if struggling with memory
        
        rotated_img2 = rotate(self.img2, rotation, reshape=False, mode='nearest')
        shifted_img2 = shift(rotated_img2, shift=(shift_y, shift_x), mode='nearest')
        
        # Plot the two images
        widths = [5, 5, 5]
        heights = [5]  
        gs_kw = dict(width_ratios=widths, height_ratios=heights)
        fig, axs = plt.subplot_mosaic([['a', 'b', 'c']], constrained_layout=True, gridspec_kw=gs_kw)
        axs['a'].imshow(self.img1, cmap='gray')
        axs['a'].set_title('Image 1 (Reference)')

        axs['b'].imshow(shifted_img2, cmap='gray')
        axs['b'].set_title(f'Image 2 (Shifted: {shift_x:.2f}, {shift_y:.2f})')

        blended_imga = self.img1
        blended_imgb = shifted_img2  
        axs['c'].imshow(blended_imga, cmap='grey', alpha=alpha)
        axs['c'].imshow(blended_imgb, cmap='magma', alpha=(1-alpha))
        axs['c'].set_title(f'Blended Image (Shift: {shift_x:.2f}, {shift_y:.2f})')
        fig.set_size_inches(12,4)
        plt.show()
        
    def on_done_button_clicked(self, b):
        plt.close('all')  
        print(f"Alignment done! Final Shift X: {self.shift_x}, Final Shift Y: {self.shift_y}, Final rotation:{self.rotation_angle}")
        self.aligned = shift(self.img2, shift=(self.shift_y, self.shift_x), mode='nearest')
        
    def manual_align(self, precision = 1):
        shift_x_slider = FloatSlider(min=-(self.img1.shape[0])/2, max=(self.img1.shape[0])/2, step=precision, value=0, description="Shift X", layout=Layout(width='1000px'), continuous_update=False)
        shift_y_slider = FloatSlider(min=-(self.img1.shape[1])/2, max=(self.img1.shape[1])/2, step=precision, value=0, description="Shift Y", layout=Layout(width='1000px'), continuous_update=False)
        alpha_slider = FloatSlider(min=0, max=1, step=0.1, value=0.5, description="Alpha", layout=Layout(width='1000px'), continuous_update=False)
        interact(
            self.plot_images, 
            shift_x=shift_x_slider,
            shift_y=shift_y_slider,
            alpha=alpha_slider
        )

        done_button = widgets.Button(description="Save Shifts", button_style='success')  # Green button
        done_button.on_click(self.on_done_button_clicked)  # Link button click to the callback
        display(done_button)

    def manual_align_with_rotation(self, precision=1, rotation_range=10):
        shift_x_slider = FloatSlider(min=-(self.img1.shape[0])/2, max=(self.img1.shape[0])/2, step=precision, value=0, description="Shift X", layout=Layout(width='1000px'), continuous_update=False)
        shift_y_slider = FloatSlider(min=-(self.img1.shape[1])/2, max=(self.img1.shape[1])/2, step=precision, value=0, description="Shift Y", layout=Layout(width='1000px'), continuous_update=False)
        rotation_slider = FloatSlider(min=-rotation_range, max=rotation_range, step=0.1, value=0, description="Rotation (°)", layout=Layout(width='1000px'), continuous_update=False)
        alpha_slider = FloatSlider(min=0, max=1, step=0.1, value=0.5, description="Alpha", layout=Layout(width='1000px'), continuous_update=False)

        interact(
            self.plot_images, 
            shift_x=shift_x_slider,
            shift_y=shift_y_slider,
            alpha=alpha_slider,
            rotation=rotation_slider
        )

        done_button = widgets.Button(description="Save Shifts + Rotation", button_style='success')
        done_button.on_click(self.on_done_button_clicked)
        display(done_button)
        
    def get_shifts(self):
        return self.shift_x, self.shift_y, self.rotation_angle


# for aligning different multiframe stacks to each other
# TO DO - UPDATE IMAGEALIGNER TO HAVE STACK ALIGNER LOGIC 
class StackAligner:
    """
    Interactive tool for manually aligning two multi-frame image stacks using
    global rigid transformations (x/y translation, z shift, and in-plane rotation).

    Input:
    Reference stacks with SAME SHAPE X,Y,Z.

    Testing new version to avoid re-plotting
    """
    def __init__(self, stack1, stack2):
        self.stack1 = stack1
        self.stack2 = stack2
        self.aligned = stack2
        self.shift_x = 0
        self.shift_y = 0
        self.shift_z = 0
        self.rotation_angle = 0
        self.frame = 0
        
    def init_view(self):
        self.fig, self.axs = plt.subplot_mosaic([['a','b','c']], figsize=(12, 4))
        self.img1 = self.axs['a'].imshow(self.stack1[0], cmap='gray')
        self.img2 = self.axs['b'].imshow(self.stack2[0], cmap='gray')
    
        self.img_overlay1 = self.axs['c'].imshow(self.stack1[0], cmap='gray', alpha=0.5)
        self.img_overlay2 = self.axs['c'].imshow(self.stack2[0], cmap='magma', alpha=0.5)
        self.axs['a'].set_title("Stack 1")
        self.axs['b'].set_title("Stack 2")
        self.axs['c'].set_title("Overlay")
        
    def redraw(self):
        f1 = self.stack1[self.frame]
    
        f2_idx = np.clip(self.frame + self.shift_z, 0, self.stack2.shape[0]-1)
        f2 = self.stack2[f2_idx]
    
        f2r = rotate(f2, self.rotation_angle, reshape=False, mode='nearest')
        f2s = shift(f2r, shift=(self.shift_y, self.shift_x), mode='nearest')
    
        self.img1.set_data(f1)
        self.img2.set_data(f2s)
    
        self.img_overlay1.set_data(f1)
        self.img_overlay2.set_data(f2s)
    
        self.fig.canvas.draw_idle()
        
    def update(self, frame, shift_x, shift_y, shift_z, rotation):
        self.frame = frame
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.shift_z = shift_z
        self.rotation_angle = rotation
    
        self.redraw()
        
    def on_done_button_clicked(self, b):
        plt.close('all')
        print(f"Alignment done! Shift X: {self.shift_x}, Shift Y: {self.shift_y}, Shift Z: {self.shift_z}, Rotation: {self.rotation_angle}")
        aligned = np.zeros_like(self.stack2)
        for i in range(self.stack2.shape[0]):
            rotated = rotate(self.stack2[i], self.rotation_angle, reshape=False, mode='nearest')
            aligned[i] = shift(rotated, shift=(self.shift_y, self.shift_x), mode='nearest')
        self.aligned = shift(aligned, shift=(self.shift_z, 0, 0), mode='nearest')
        
        cwd = os.getcwd()
        filename = os.path.join(cwd, 'aligned_file_temp.npz')
        np.savez(
            filename,
            aligned=self.aligned,
            shift_x=np.array(self.shift_x),
            shift_y=np.array(self.shift_y),
            shift_z=np.array(self.shift_z),
            rotation=np.array(self.rotation_angle)
        ) 
        print(f'saved data in temp file {filename}. test and rename accordingly.')
        
    def manual_align(self, precision=1):
        self.init_view() 
        n_frames = self.stack1.shape[0]
        frame_slider   = IntSlider(min=0, max=n_frames - 1, step=1, value=0, description="Frame",   layout=Layout(width='1000px'), continuous_update=False)
        shift_x_slider = FloatSlider(min=-(self.stack1.shape[2])/2, max=(self.stack1.shape[2])/2, step=precision, value=0, description="Shift X", layout=Layout(width='1000px'), continuous_update=False)
        shift_y_slider = FloatSlider(min=-(self.stack1.shape[1])/2, max=(self.stack1.shape[1])/2, step=precision, value=0, description="Shift Y", layout=Layout(width='1000px'), continuous_update=False)
        shift_z_slider = IntSlider(min=-(n_frames - 1), max=(n_frames - 1), step=1, value=0, description="Shift Z", layout=Layout(width='1000px'), continuous_update=False)
        alpha_slider   = FloatSlider(min=0, max=1, step=0.1, value=0.5, description="Alpha",   layout=Layout(width='1000px'), continuous_update=False)
        interact(
            self.update,
            frame=frame_slider,
            shift_x=shift_x_slider,
            shift_y=shift_y_slider,
            shift_z=shift_z_slider,
            alpha=alpha_slider
        )
        done_button = widgets.Button(description="Save Shifts", button_style='success')
        done_button.on_click(self.on_done_button_clicked)
        display(done_button)
 
    def manual_align_with_rotation(self, precision=1, rotation_range=10):
        self.init_view() 
        n_frames = self.stack1.shape[0]
        frame_slider    = IntSlider(min=0, max=n_frames - 1, step=1, value=0, description="Frame",       layout=Layout(width='1000px'), continuous_update=False)
        shift_x_slider  = FloatSlider(min=-(self.stack1.shape[2])/2, max=(self.stack1.shape[2])/2, step=precision, value=0, description="Shift X",   layout=Layout(width='1000px'), continuous_update=False)
        shift_y_slider  = FloatSlider(min=-(self.stack1.shape[1])/2, max=(self.stack1.shape[1])/2, step=precision, value=0, description="Shift Y",   layout=Layout(width='1000px'), continuous_update=False)
        shift_z_slider  = IntSlider(min=-(n_frames - 1), max=(n_frames - 1), step=1, value=0, description="Shift Z",   layout=Layout(width='1000px'), continuous_update=False)
        rotation_slider = FloatSlider(min=-rotation_range, max=rotation_range, step=0.1, value=0, description="Rotation (°)", layout=Layout(width='1000px'), continuous_update=False)
        alpha_slider    = FloatSlider(min=0, max=1, step=0.1, value=0.5, description="Alpha",     layout=Layout(width='1000px'), continuous_update=False)
        interact(
            self.update,
            frame=frame_slider,
            shift_x=shift_x_slider,
            shift_y=shift_y_slider,
            shift_z=shift_z_slider,
            alpha=alpha_slider,
            rotation=rotation_slider
        )
        done_button = widgets.Button(description="Save Shifts + Rotation", button_style='success')
        done_button.on_click(self.on_done_button_clicked)
        display(done_button)
 
    def get_shifts(self):
        return self.shift_x, self.shift_y, self.shift_z, self.rotation_angle
