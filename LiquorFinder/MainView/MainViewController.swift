//
//  MainViewController.swift
//  LiquorFinder
//
//  Created by Jinsan Kim on 2022/08/29.
//

import UIKit
import AVFoundation
import Vision

class MainViewController: UIViewController {
    let imagePredictor = ImagePredictor()
    let predictionsToShow = 2
    var firstRun = true
    
    private let cameraButton: UIButton = {
      let button = UIButton()
      button.setTitle("camera", for: .normal)
      button.setTitleColor(.systemBlue, for: .normal)
      button.setTitleColor(.blue, for: .highlighted)
      button.addTarget(self, action: #selector(openCamera), for: .touchUpInside)
      return button
    }()
    
    private let imageView: UIImageView = {
      let view = UIImageView()
      return view
    }()
    
    lazy var predictionLabel: UILabel = {
        let label = UILabel()
        label.textColor = .label
        label.text = "???"
        return label
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let screenHeight = self.view.frame.height
        
        self.view.addSubview(imageView)
        self.view.addSubview(cameraButton)
        self.view.addSubview(predictionLabel)
        setImageView(height: screenHeight)
        setCameraButton()
        setPredictionLabel()
        
        guard let image = imageView.image else { return }
        userSelectedPhoto(image)
    }
    
    func showAlertGoToSetting() {
        let alertController = UIAlertController(
          title: "현재 카메라 사용에 대한 접근 권한이 없습니다.",
          message: "설정 > {앱 이름}탭에서 접근을 활성화 할 수 있습니다.",
          preferredStyle: .alert
        )
        let cancelAlert = UIAlertAction(
          title: "취소",
          style: .cancel
        ) { _ in
            alertController.dismiss(animated: true, completion: nil)
          }
        let goToSettingAlert = UIAlertAction(
          title: "설정으로 이동하기",
          style: .default) { _ in
            guard
              let settingURL = URL(string: UIApplication.openSettingsURLString),
              UIApplication.shared.canOpenURL(settingURL)
            else { return }
            UIApplication.shared.open(settingURL, options: [:])
          }
        [cancelAlert, goToSettingAlert]
          .forEach(alertController.addAction(_:))
        DispatchQueue.main.async {
          self.present(alertController, animated: true) // must be used from main thread only
        }
      }
    
    @objc private func openCamera() {
        #if targetEnvironment(simulator)
        fatalError()
        #endif
        
////         Privacy - Camera Usage Description
//        AVCaptureDevice.requestAccess(for: .video) { [weak self] isAuthorized in
//            guard isAuthorized else {
//                self?.showAlertGoToSetting() // 밑에서 계속 구현
//                return
//            }
//
////         TODO - 카메라 열기
//            DispatchQueue.main.async {
//                let pickerController = UIImagePickerController() // must be used from main thread only
//                pickerController.sourceType = .camera
//                pickerController.allowsEditing = false
//                pickerController.mediaTypes = ["public.image"]
//    //             만약 비디오가 필요한 경우,
//    //            imagePicker.mediaTypes = ["public.movie"]
//    //            imagePicker.videoQuality = .typeHigh
//                pickerController.delegate = self
//            }
//        }
        
        present(cameraPicker, animated: false)
    }
    
    func setImageView(height: CGFloat) {
        imageView.translatesAutoresizingMaskIntoConstraints = false
        self.imageView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.centerYAnchor, constant: -height/4).isActive = true
        self.imageView.leftAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leftAnchor, constant: 10).isActive = true
        self.imageView.rightAnchor.constraint(equalTo: view.safeAreaLayoutGuide.rightAnchor, constant: -10).isActive = true
        self.imageView.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -height/4).isActive = true
    }
    
    func setCameraButton() {
        cameraButton.translatesAutoresizingMaskIntoConstraints = false
        cameraButton.centerXAnchor.constraint(equalTo: view.safeAreaLayoutGuide.centerXAnchor).isActive = true
        cameraButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -50).isActive = true
    }
    
    func setPredictionLabel() {
        predictionLabel.translatesAutoresizingMaskIntoConstraints = false
        predictionLabel.topAnchor.constraint(equalTo: cameraButton.safeAreaLayoutGuide.topAnchor, constant: -50).isActive = true
        predictionLabel.centerXAnchor.constraint(equalTo: view.safeAreaLayoutGuide.centerXAnchor).isActive = true
    }
}

extension MainViewController {
//     MARK: Main storyboard updates
    /// Updates the storyboard's image view.
    /// - Parameter image: An image.
    func updateImage(_ image: UIImage) {
        DispatchQueue.main.async {
            self.imageView.image = image
        }
    }

    /// Updates the storyboard's prediction label.
    /// - Parameter message: A prediction or message string.
    /// - Tag: updatePredictionLabel
    func updatePredictionLabel(_ message: String) {
        DispatchQueue.main.async {
            self.predictionLabel.text = message
        }

        if firstRun {
            DispatchQueue.main.async {
                self.firstRun = false
                self.predictionLabel.superview?.isHidden = false
            }
        }
    }
    /// Notifies the view controller when a user selects a photo in the camera picker or photo library picker.
    /// - Parameter photo: A photo from the camera or photo library.
    func userSelectedPhoto(_ photo: UIImage) {
        updateImage(photo)
        updatePredictionLabel("Making predictions for the photo...")

        DispatchQueue.global(qos: .userInitiated).async {
            self.classifyImage(photo)
        }
    }

}

extension MainViewController {
    // MARK: Image prediction methods
    /// Sends a photo to the Image Predictor to get a prediction of its content.
    /// - Parameter image: A photo.
    private func classifyImage(_ image: UIImage) {
        do {
            try self.imagePredictor.makePredictions(for: image,
                                                    completionHandler: imagePredictionHandler)
        } catch {
            print("Vision was unable to make a prediction...\n\n\(error.localizedDescription)")
        }
    }

    /// The method the Image Predictor calls when its image classifier model generates a prediction.
    /// - Parameter predictions: An array of predictions.
    /// - Tag: imagePredictionHandler
    private func imagePredictionHandler(_ predictions: [ImagePredictor.Prediction]?) {
        guard let predictions = predictions else {
            updatePredictionLabel("No predictions. (Check console log.)")
            return
        }

        let formattedPredictions = formatPredictions(predictions)

        let predictionString = formattedPredictions.joined(separator: "\n")
        updatePredictionLabel(predictionString)
    }

    /// Converts a prediction's observations into human-readable strings.
    /// - Parameter observations: The classification observations from a Vision request.
    /// - Tag: formatPredictions
    private func formatPredictions(_ predictions: [ImagePredictor.Prediction]) -> [String] {
        // Vision sorts the classifications in descending confidence order.
        let topPredictions: [String] = predictions.prefix(predictionsToShow).map { prediction in
            var name = prediction.classification

            // For classifications with more than one name, keep the one before the first comma.
            if let firstComma = name.firstIndex(of: ",") {
                name = String(name.prefix(upTo: firstComma))
            }

            return "\(name) - \(prediction.confidencePercentage)%"
        }

        return topPredictions
    }
}

