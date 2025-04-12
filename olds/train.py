import torch
from model import SSDResNetFaceLandmarkGaze
# Training function
def train_model(model, train_loader, optimizer, criterion, device, num_epochs=10):
    """
    Train the SSD model
    """
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        face_loss = 0.0
        landmark_loss = 0.0
        gaze_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            
            # Move targets to device
            for key in targets:
                if isinstance(targets[key], torch.Tensor):
                    targets[key] = targets[key].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(images)
            
            # Compute loss
            loss_dict = criterion(predictions, targets)
            
            # Backward pass
            loss = loss_dict['total_loss']
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Update statistics
            epoch_loss += loss.item()
            face_loss += loss_dict['face_loss'].item()
            landmark_loss += loss_dict['landmark_loss'].item()
            gaze_loss += loss_dict['gaze_loss'].item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        # Print epoch statistics
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}, '
              f'Face Loss: {face_loss/len(train_loader):.4f}, '
              f'Landmark Loss: {landmark_loss/len(train_loader):.4f}, '
              f'Gaze Loss: {gaze_loss/len(train_loader):.4f}')

# Example usage:
if __name__ == "__main__":
    # Create model
    model = SSDResNetFaceLandmarkGaze(num_classes=2, image_size=300)
    
    # Sample input
    x = torch.randn(2, 3, 300, 300)
    
    # Forward pass
    output = model(x)
    
    # Print output shapes
    print(f"Face predictions shape: {output['face_preds'].shape}")
    print(f"Landmark predictions shape: {output['landmark_preds'].shape}")
    print(f"Gaze predictions shape: {output['gaze_preds'].shape}")
    print(f"Anchors shape: {output['anchors'].shape}")