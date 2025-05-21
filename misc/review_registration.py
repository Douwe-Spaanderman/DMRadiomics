import os
import sys
import json
import nibabel as nib
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QCheckBox, QPushButton, QFileDialog, QScrollArea, QSlider, QComboBox
)
from PyQt5.QtCore import Qt, QEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from skimage import measure

class SequencePanel(QWidget):
    def __init__(self, name, image, rigid_mask, affine_mask, is_reference=False):
        super().__init__()
        self.name = name
        self.image = image
        self.rigid_mask = rigid_mask
        self.affine_mask = affine_mask
        self.is_reference = is_reference
        self.current_mask_type = "rigid"

        self.setFocusPolicy(Qt.StrongFocus)
        self.installEventFilter(self)

        self.current_slice = self.initial_slice_from_mask()

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.canvas = FigureCanvas(plt.figure(figsize=(3, 3), facecolor='black'))
        self.canvas.setFixedSize(512, 512)
        self.ax = self.canvas.figure.add_subplot(111)
        self.layout.addWidget(self.canvas)

        self.label = QLabel(f"{name} {'(reference)' if is_reference else ''}")
        self.layout.addWidget(self.label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(image.shape[2] - 1)
        self.slider.setValue(self.current_slice)
        self.slider.valueChanged.connect(self.slider_changed)
        self.layout.addWidget(self.slider)

        self.checkbox = QCheckBox("Registration OK?")
        if is_reference:
            self.checkbox.setEnabled(False)
        self.layout.addWidget(self.checkbox)

        if not is_reference:
            self.toggle_button = QPushButton("Using: Rigid")
            self.toggle_button.clicked.connect(self.toggle_mask)
            self.layout.addWidget(self.toggle_button)

        self.draw()

    def initial_slice_from_mask(self):
        mask = self.rigid_mask if self.rigid_mask is not None else self.affine_mask
        if mask is not None and np.any(mask):
            z_nonzero = np.any(np.any(mask, axis=0), axis=0)
            z_indices = np.where(z_nonzero)[0]
            if len(z_indices) > 0:
                return int(np.median(z_indices))
        return self.image.shape[2] // 2

    def toggle_mask(self):
        self.current_mask_type = "affine" if self.current_mask_type == "rigid" else "rigid"
        self.toggle_button.setText(f"Using: {self.current_mask_type.capitalize()}")
        self.draw()

    def draw(self):
        self.ax.clear()
        self.ax.axis("off")
        try:
            self.ax.imshow(self.image[:, :, self.current_slice], cmap="gray")
        except IndexError as e:
            print(f"Error displaying the correct imag slice: {e}")
            self.ax.imshow(self.image[:, :, 0], cmap="gray")

        rigid_slice = self.rigid_mask[:, :, self.current_slice] if self.rigid_mask is not None else None
        affine_slice = self.affine_mask[:, :, self.current_slice] if self.affine_mask is not None else None

        if self.current_mask_type == "rigid":
            if affine_slice is not None and np.any(affine_slice):
                contours = measure.find_contours(affine_slice, 0.5)
                for contour in contours:
                    self.ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color="blue", alpha=0.5)
            if rigid_slice is not None and np.any(rigid_slice):
                contours = measure.find_contours(rigid_slice, 0.5)
                for contour in contours:
                    self.ax.plot(contour[:, 1], contour[:, 0], linewidth=2.0, color="red", alpha=0.9)
        else:
            if rigid_slice is not None and np.any(rigid_slice):
                contours = measure.find_contours(rigid_slice, 0.5)
                for contour in contours:
                    self.ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color="red", alpha=0.5)
            if affine_slice is not None and np.any(affine_slice):
                contours = measure.find_contours(affine_slice, 0.5)
                for contour in contours:
                    self.ax.plot(contour[:, 1], contour[:, 0], linewidth=2.0, color="blue", alpha=0.9)

        self.canvas.draw()

    def update_slice(self, z):
        self.current_slice = z
        self.slider.blockSignals(True)
        self.slider.setValue(z)
        self.slider.blockSignals(False)
        self.draw()

    def slider_changed(self, value):
        self.current_slice = value
        self.draw()

    def eventFilter(self, source, event):
        if event.type() == QEvent.Wheel:
            delta = event.angleDelta().y()
            self.scroll_slice(delta > 0)
            return True
        elif event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Up:
                self.scroll_slice(True)
                return True
            elif event.key() == Qt.Key_Down:
                self.scroll_slice(False)
                return True
        return False

    def scroll_slice(self, forward):
        if forward:
            self.current_slice = min(self.current_slice + 1, self.image.shape[2] - 1)
        else:
            self.current_slice = max(self.current_slice - 1, 0)
        self.update_slice(self.current_slice)


class PatientViewer(QWidget):
    def __init__(self, root_folder):
        super().__init__()
        self.setWindowTitle("Patient Registration Viewer")
        self.root_folder = root_folder
        self.patient_dirs = sorted([
            os.path.join(root_folder, d)
            for d in os.listdir(root_folder)
            if os.path.isdir(os.path.join(root_folder, d))
        ])
        self.current_psatient_index = 0
        for idx, patient_dir in enumerate(self.patient_dirs):
            review_path = os.path.join(patient_dir, 'registration_review.json')
            if not os.path.exists(review_path):
                self.current_patient_index = idx
                break
        else:
            self.current_patient_index = len(self.patient_dirs)
            print("All patients already reviewed.")

        self.sequence_panels = []
        self.patient_cache = {}

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.scroll_area = QScrollArea()
        self.scroll_widget = QWidget()
        self.scroll_layout = QHBoxLayout()
        self.scroll_widget.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_widget)
        self.scroll_area.setWidgetResizable(True)
        self.main_layout.addWidget(self.scroll_area)

        btn_layout = QHBoxLayout()
        self.save_button = QPushButton("Save (s)")
        self.save_button.clicked.connect(self.save_results)
        self.prev_button = QPushButton("Previous (p)")
        self.prev_button.clicked.connect(self.load_previous_patient)
        self.next_button = QPushButton("Next (n)")
        self.next_button.clicked.connect(self.load_next_patient)
        btn_layout.addWidget(self.save_button)
        btn_layout.addWidget(self.prev_button)
        btn_layout.addWidget(self.next_button)
        self.main_layout.addLayout(btn_layout)

        self.load_patient()

    def load_patient(self):
        # Clear previous
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        self.sequence_panels.clear()

        patient_folder = self.patient_dirs[self.current_patient_index]
        self.setWindowTitle(f"Patient Viewer - {os.path.basename(patient_folder)}")
        files = sorted(os.listdir(patient_folder))
        images = [f for f in files if f.endswith(".nii.gz") and "-mask" not in f]

        ref_image_name = None
        for img in images:
            if img.replace(".nii.gz", "-mask.nii.gz") in files:
                ref_image_name = img
                break

        if not ref_image_name:
            print(f"No reference image found in {patient_folder}")
            return

        ref_img = nib.load(os.path.join(patient_folder, ref_image_name)).get_fdata()
        ref_mask = nib.load(os.path.join(patient_folder, ref_image_name.replace(".nii.gz", "-mask.nii.gz"))).get_fdata()
        panel = SequencePanel(ref_image_name, ref_img, ref_mask, None, is_reference=True)
        self.sequence_panels.append(panel)
        self.scroll_layout.addWidget(panel)

        for img in images:
            if img == ref_image_name:
                continue
            rigid_mask_path = os.path.join(patient_folder, img.replace(".nii.gz", "-rigid-mask.nii.gz"))
            affine_mask_path = os.path.join(patient_folder, img.replace(".nii.gz", "-affine-mask.nii.gz"))
            if not (os.path.exists(rigid_mask_path) and os.path.exists(affine_mask_path)):
                continue
            image_data = nib.load(os.path.join(patient_folder, img)).get_fdata()
            rigid_mask = nib.load(rigid_mask_path).get_fdata()
            affine_mask = nib.load(affine_mask_path).get_fdata()
            panel = SequencePanel(img, image_data, rigid_mask, affine_mask)
            self.sequence_panels.append(panel)
            self.scroll_layout.addWidget(panel)

        self.load_results()

    def save_results(self):
        patient_folder = self.patient_dirs[self.current_patient_index]
        results = {}
        for panel in self.sequence_panels:
            if not panel.is_reference:
                results[panel.name] = {
                    "RegistrationOK": panel.checkbox.isChecked(),
                    "ChosenMask": panel.current_mask_type
                }
        json_path = os.path.join(patient_folder, "registration_review.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {json_path}")

    def load_results(self):
        patient_folder = self.patient_dirs[self.current_patient_index]
        json_path = os.path.join(patient_folder, "registration_review.json")
        if not os.path.exists(json_path):
            return
        with open(json_path, "r") as f:
            results = json.load(f)
        for panel in self.sequence_panels:
            if not panel.is_reference and panel.name in results:
                panel.checkbox.setChecked(results[panel.name]["RegistrationOK"])
                panel.current_mask_type = results[panel.name].get("ChosenMask", "rigid")
                if hasattr(panel, "toggle_button"):
                    panel.toggle_button.setText(f"Using: {panel.current_mask_type.capitalize()}")
                panel.draw()

    def load_next_patient(self):
        self.save_results()
        if self.current_patient_index < len(self.patient_dirs) - 1:
            self.current_patient_index += 1
            self.load_patient()

    def load_previous_patient(self):
        self.save_results()
        if self.current_patient_index > 0:
            self.current_patient_index -= 1
            self.load_patient()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_N:
            self.load_next_patient()
        elif key == Qt.Key_P:
            self.load_previous_patient()
        elif key == Qt.Key_S:
            self.save_results()
        elif Qt.Key_1 <= key <= Qt.Key_9:
            index = key - Qt.Key_1
            if index < len(self.sequence_panels):
                panel = self.sequence_panels[index]
                if not panel.is_reference:
                    panel.checkbox.setChecked(not panel.checkbox.isChecked())
        elif key == Qt.Key_Left:
            self.scroll_area.horizontalScrollBar().setValue(
                self.scroll_area.horizontalScrollBar().value() - 100
            )
        elif key == Qt.Key_Right:
            self.scroll_area.horizontalScrollBar().setValue(
                self.scroll_area.horizontalScrollBar().value() + 100
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    root = QFileDialog.getExistingDirectory(None, "Select Root Folder With Patients")
    if root:
        viewer = PatientViewer(root)
        viewer.resize(1200, 800)
        viewer.show()
        sys.exit(app.exec_())
