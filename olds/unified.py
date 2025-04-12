import cv2

class UnifiedDetectionPipeline:
    """
    Unified pipeline for face, gaze, landmark and phone detection
    """
    def __init__(self, model_path: str = None, device: str = None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the unified SSD model
        self.model = SSDResNetFaceLandmarkGaze(
            num_classes=3,  # Background + Face + Phone
            image_size=300
        ).to(self.device)
        
        # Load pretrained weights if provided
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.eval()
        
        # Define image transforms
        self.transforms = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image):
        """Preprocess image for model input"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Apply transforms
        image = self.transforms(image).unsqueeze(0)
        return image.to(self.device)

    def postprocess_results(self, predictions, confidence_threshold=0.5):
        """Postprocess model predictions"""
        results = []
        
        for pred in predictions:
            boxes = pred['boxes']
            scores = pred['scores']
            landmarks = pred['landmarks']
            gaze = pred['gaze']
            
            # Separate face and phone detections based on class predictions
            face_mask = scores > confidence_threshold
            face_boxes = boxes[face_mask]
            face_landmarks = landmarks[face_mask]
            face_gaze = gaze[face_mask]
            
            # Convert predictions to desired format
            processed_results = {
                'faces': [],
                'phones': []
            }
            
            # Process face detections
            for box, lm, gz in zip(face_boxes, face_landmarks, face_gaze):
                face_dict = {
                    'bbox': box.cpu().numpy(),
                    'landmarks': lm.reshape(-1, 2).cpu().numpy(),
                    'gaze': {
                        'pitch': gz[0].item(),
                        'yaw': gz[1].item()
                    }
                }
                processed_results['faces'].append(face_dict)
            
            results.append(processed_results)
        
        return results

    def __call__(self, image):
        """
        Run inference on an image
        
        Args:
            image: numpy array (BGR) or PIL Image
            
        Returns:
            Dictionary containing:
                - faces: List of face detections with landmarks and gaze
                - phones: List of phone detections
        """
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model.predict(
                input_tensor,
                confidence_threshold=0.5,
                nms_threshold=0.5
            )
        
        # Postprocess results
        results = self.postprocess_results(predictions)
        
        return results[0]  # Return results for single image

# Modified SSDHead to include phone detection
class SSDHead(nn.Module):
    """
    Modified SSD detection head to include phone detection
    """
    def __init__(self, in_channels: int, num_anchors: int):
        super(SSDHead, self).__init__()
        # Face and phone detection: 4 box coordinates + 2 class scores (face, phone)
        self.det_conv = nn.Conv2d(in_channels, num_anchors * 6, kernel_size=3, padding=1)
        
        # Facial landmarks: 68 landmarks with x,y coordinates = 136 values
        self.landmarks_conv = nn.Conv2d(in_channels, num_anchors * 136, kernel_size=3, padding=1)
        
        # Gaze estimation: pitch and yaw angles (2 values)
        self.gaze_conv = nn.Conv2d(in_channels, num_anchors * 2, kernel_size=3, padding=1)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Detection (face and phone)
        det_preds = self.det_conv(x)
        det_preds = det_preds.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 6)
        
        # Split detection predictions
        box_preds = det_preds[..., :4]
        class_preds = det_preds[..., 4:]  # 2 classes: face and phone
        
        # Landmark detection (only for faces)
        landmark_preds = self.landmarks_conv(x)
        landmark_preds = landmark_preds.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 136)
        
        # Gaze estimation (only for faces)
        gaze_preds = self.gaze_conv(x)
        gaze_preds = gaze_preds.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
        
        return box_preds, class_preds, landmark_preds, gaze_preds

# Example usage:
def main():
    # Initialize pipeline
    pipeline = UnifiedDetectionPipeline()
    
    # Load and process image
    image = cv2.imread('example.jpg')
    results = pipeline(image)
    
    # Process results
    for face in results['faces']:
        bbox = face['bbox']
        landmarks = face['landmarks']
        gaze = face['gaze']
        
        # Draw face bbox
        cv2.rectangle(image, 
                     (int(bbox[0]), int(bbox[1])), 
                     (int(bbox[2]), int(bbox[3])), 
                     (0, 255, 0), 2)
        
        # Draw landmarks
        for (x, y) in landmarks:
            cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)
        
        # Draw gaze direction
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Convert gaze angles to vector
        pitch = gaze['pitch']
        yaw = gaze['yaw']
        
        # Draw gaze vector
        length = 50
        dx = -length * np.sin(yaw) * np.cos(pitch)
        dy = -length * np.sin(pitch)
        
        cv2.line(image,
                 (int(center_x), int(center_y)),
                 (int(center_x + dx), int(center_y + dy)),
                 (255, 0, 0), 2)
    
    # Display results
    cv2.imshow('Results', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()