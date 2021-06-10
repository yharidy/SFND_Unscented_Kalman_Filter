#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  
  // initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  // set state dimension and augmented state dimension
  n_x_ = 5;
  n_aug_ = 7;

  // set lambda
  lambda_ = 3 - n_x_;

  // initialize state timestamp
  time_us_ = 0.0;

  // initialize covariance matrix
  //P_ = Eigen::MatrixXd::Identity(n_x_, n_x_);
  
  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1000, 0, 0,
        0, 0, 0, 0.0225, 0,
        0, 0, 0, 0, 0.0225;
  // set weights
  weights_ = VectorXd(2*n_aug_+1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  for (int i = 1; i < 2*n_aug_+1; ++i) {
    weights_(i) = 0.5 / (lambda_ + n_aug_);
  }

}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if (!is_initialized_) {
    if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER) {
      x_(0) = meas_package.raw_measurements_(0);
      x_(1) = meas_package.raw_measurements_(1);
      x_(2) = 0.0;
      x_(3) = 0.0;
      x_(4) = 0.0;           
    }
    else {
      double roh = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      double rohd = meas_package.raw_measurements_(2);
      double vx = rohd * cos(phi);
      double vy = rohd * sin(phi);
      double v = sqrt(vx * vx + vy * vy);

      x_(0) = roh * cos(phi);
      x_(1) = roh * sin(phi);
      x_(2) = v;
      x_(3) = 0.0;
      x_(4) = 0.0;
    }
    is_initialized_ = true;
    time_us_ = meas_package.timestamp_;
    return;
  }
  
  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0 ;

  UKF::Prediction(delta_t);  
  
  if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER && use_laser_) {
    UKF::UpdateLidar(meas_package); 
  }
  if (meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR && use_radar_) {
    UKF::UpdateRadar(meas_package); 
  }

  time_us_ = meas_package.timestamp_;
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

  // --------------------- Generate sigma points
  // augment state vector and covariance matrix
  Eigen::VectorXd xaug = VectorXd(n_aug_);
  xaug.head(n_x_) = x_;
  xaug(n_x_) = 0.0;
  xaug(n_x_+1) = 0.0;
  
  MatrixXd Paug = MatrixXd(n_aug_, n_aug_);
  Paug.fill(0.0);
  Paug.topLeftCorner(n_x_, n_x_) = P_;
  Paug(5,5) = std_a_ * std_a_;
  Paug(6,6) = std_yawdd_ * std_yawdd_;
  
  MatrixXd Xsig = MatrixXd(n_aug_,2*n_aug_+1);
  // set first column of sigma point matrix
  Xsig.col(0) = xaug;
  
  // calculate square root of P
  MatrixXd A = Paug.llt().matrixL();
  // set remaining sigma points
  for (int i = 0; i < n_aug_; ++i) {
    Xsig.col(i+1)     = xaug + sqrt(lambda_+n_aug_) * A.col(i);
    Xsig.col(i+n_aug_+1) = xaug - sqrt(lambda_+n_aug_) * A.col(i);
  }
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);
  // --------------------- Predict sigma points
  for (int i = 0; i< 2*n_aug_+1; ++i) {
    
    double p_x = Xsig(0,i);
    double p_y = Xsig(1,i);
    double v = Xsig(2,i);
    double yaw = Xsig(3,i);
    double yawd = Xsig(4,i);
    double nu_a = Xsig(5,i);
    double nu_yawdd = Xsig(6,i);

    // predicted state values
    double px_p, py_p;

    // avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v * (sin(yaw + yawd*delta_t) - sin(yaw)) / yawd;
        py_p = p_y + v * (cos(yaw) - cos(yaw+yawd*delta_t)) / yawd;
    } else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    Xsig_pred_(0,i) = px_p + 0.5 * delta_t * delta_t * nu_a * cos(yaw);
    Xsig_pred_(1,i) = py_p + 0.5 * delta_t * delta_t * nu_a * sin(yaw);
    Xsig_pred_(2,i) = v + nu_a * delta_t;
    Xsig_pred_(3,i) = yaw + yawd * delta_t + 0.5 * delta_t * delta_t * nu_yawdd;
    Xsig_pred_(4,i) = yawd + nu_yawdd * delta_t;
    
  }

  // --------------------- Predicted mean state and covariance matrix
  //create mean state and covariance matrix
  VectorXd Xmean = VectorXd(n_x_);
  Xmean.fill(0.0);
  MatrixXd P_pred = MatrixXd(n_x_, n_x_);
  P_pred.fill(0.0);

  // fill Xmean
  for (int i = 0; i < 2*n_aug_+1; ++i) {
    Xmean = Xmean + weights_(i) * Xsig_pred_.col(i);
  }

  // fill P_pred
  for (int i = 0; i < 2*n_aug_+1; ++i) {
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - Xmean;
    // angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_pred = P_pred + weights_(i) * x_diff * x_diff.transpose();
  }

  // Update state and covariance
  x_ = Xmean;
  P_ = P_pred;
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  
  // measurement matrix
  MatrixXd H = MatrixXd(2,n_x_);

  // measurement covariance matrix
  MatrixXd R = MatrixXd(2,2);

  // Fille H and R
  H.fill(0.0);
  H(0,0) = 1.0;
  H(1,1) = 1.0;
  R << std_laspx_ * std_laspx_, 0,
       0, std_laspy_ * std_laspy_;

  VectorXd z_pred = H * x_;
  VectorXd z = VectorXd(2);
  z(0) = meas_package.raw_measurements_(0);
  z(1) = meas_package.raw_measurements_(1);
  VectorXd y = z - z_pred;
  MatrixXd Ht = H.transpose();
  MatrixXd S = H * P_ * Ht + R;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  MatrixXd I = MatrixXd::Identity(n_x_, n_x_);
  P_ = (I - K * H) * P_;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  // Get measurement data
  int n_z = 3; // size of measurement vector
  VectorXd z = VectorXd(n_z);
  z = meas_package.raw_measurements_;
  // --------------------- Project predicted sigma points into measurement space
  MatrixXd Zsig_pred = MatrixXd(n_z,2*n_aug_+1);

  for (int i = 0; i < 2*n_aug_+1; ++i) {
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);
    double yawd = Xsig_pred_(4,i);

    double roh = sqrt(p_x * p_x + p_y * p_y);
    double phi = atan2(p_y, p_x);
    double rohd = (p_x*v*cos(yaw) + p_y*v*sin(yaw)) / std::max(0.0001, roh);

    Zsig_pred(0,i) = roh;
    Zsig_pred(1,i) = phi;
    Zsig_pred(2,i) = rohd;
  } 

  // --------------------- Get predicted measurement mean and predicted covariace matrix S
  VectorXd Zmean_pred = VectorXd(n_z);
  Zmean_pred.fill(0.0);

  // fill Zmean_pred
  for (int i = 0; i < 2*n_aug_+1; ++i) {
    Zmean_pred = Zmean_pred + weights_(i) * Zsig_pred.col(i);
  }

  // create predicted covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);

  // create measurement noise matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R.fill(0.0);
  R(0,0) = std_radr_ * std_radr_;
  R(1,1) = std_radphi_ * std_radphi_;
  R(2,2) = std_radrd_ * std_radrd_;

  // fill S
  for (int i = 0; i < 2*n_aug_+1; ++i) {
    
    VectorXd z_diff = Zsig_pred.col(i) - Zmean_pred;

    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  S = S + R;

  // --------------------- Update measurement
  // create cross-correlation matrix 
  MatrixXd T = MatrixXd(n_x_,n_z);
  T.fill(0.0);

  for (int i = 0; i < 2*n_aug_+1; ++i) {
    
    VectorXd z_diff = Zsig_pred.col(i) - Zmean_pred;
    
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    
    while (x_diff(3)> M_PI) x_diff(1)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(1)+=2.*M_PI;
      
    T = T + weights_(i) * x_diff * z_diff.transpose();
  }
  
  // calculate Kalman gain 
  MatrixXd K = T * S.inverse();

  // Update state with new measurement 
  VectorXd z_diff = z - Zmean_pred;
  
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

}