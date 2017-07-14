/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <random>
#include <algorithm>
#include <numeric>
#include <math.h> 
#include <sstream>
#include <iterator>
#include <functional>


namespace
{
	const int NUMBER_PARTICLES = 1000;
	const double EPSILON = 1e-4;

	std::vector<LandmarkObs> transformObservation(Particle& particle, const std::vector<LandmarkObs>& observations) {
		std::vector<LandmarkObs> trObs;
		particle.associations.clear();
		particle.sense_x.clear();
		particle.sense_y.clear();

		for(const auto& obs: observations) {
			LandmarkObs newObs;
			newObs.x = particle.x + obs.x*cos(particle.theta) - obs.y*sin(particle.theta);
			newObs.y = particle.y + obs.x*sin(particle.theta) + obs.y*cos(particle.theta);
			trObs.push_back(newObs);
			particle.sense_x.push_back(newObs.x);
			particle.sense_y.push_back(newObs.y);
		}
		return trObs;
	}

	std::vector<LandmarkObs> predictLandmarks(const Particle& particle, const Map& map, const double sensor_range) {
		std::vector<LandmarkObs> preds;
		for(const auto& landmark: map.landmark_list) {
			if(dist(landmark.x_f, landmark.y_f, particle.x, particle.y) < sensor_range) {
				LandmarkObs predLandmark;
				predLandmark.id = landmark.id_i;
				predLandmark.x = landmark.x_f;
				predLandmark.y = landmark.y_f;
				preds.push_back(predLandmark);
			}
		}
		return preds;
	}

	double multGaussProb(const double x, const double y, const double mu_x, const double mu_y, const double std[]) {
		return exp(-(x - mu_x)*(x - mu_x) / (2 * std[0] * std[0]) - (y - mu_y)*(y - mu_y) / (2 * std[1] * std[1])) / (2 * M_PI*std[0] * std[1]);
	}
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = NUMBER_PARTICLES;
	//weights = std::vector<double>(num_particles, 1.);
	particles.resize(num_particles);

	std::default_random_engine gen;
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);

	for(int i = 0; num_particles > i; ++i) {
		auto& particle = particles[i];
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1.;
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	std::function<void(Particle&)> pred;
	std::default_random_engine gen;

	if(std::abs(yaw_rate) < EPSILON) {
		pred = [&](Particle& particle){
			particle.x += velocity*delta_t*cos(particle.theta);
			particle.y += velocity*delta_t*sin(particle.theta);
			std::normal_distribution<double> dist_x(particle.x, std_pos[0]);
			std::normal_distribution<double> dist_y(particle.y, std_pos[1]);
			std::normal_distribution<double> dist_theta(particle.theta, std_pos[2]);
			particle.x = dist_x(gen);
			particle.y = dist_y(gen);
			particle.theta = dist_theta(gen);
		};
	}else {
		pred = [&](Particle& particle) {
			particle.x += velocity*(sin(particle.theta + yaw_rate*delta_t) - sin(particle.theta)) / yaw_rate;
			particle.y += velocity*(cos(particle.theta) - cos(particle.theta + yaw_rate*delta_t)) / yaw_rate;
			particle.theta += yaw_rate*delta_t;
			std::normal_distribution<double> dist_x(particle.x, std_pos[0]);
			std::normal_distribution<double> dist_y(particle.y, std_pos[1]);
			std::normal_distribution<double> dist_theta(particle.theta, std_pos[2]);
			particle.x = dist_x(gen);
			particle.y = dist_y(gen);
			particle.theta = dist_theta(gen);
		};
	}
	std::for_each(particles.begin(), particles.end(), pred);
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	const size_t numberPred = predicted.size();
	for(auto& obs: observations) {
		double minDist = std::numeric_limits<double>::max();
		for (size_t i = 0; numberPred > i; ++i) {
			const auto& landmark = predicted[i];
			const double currentDist = dist(obs.x, obs.y, landmark.x, landmark.y);
			if(minDist > currentDist) {
				minDist = currentDist;
				obs.id = static_cast<int>(i);
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	double sumWeights(0);
	weights.clear();
	for(auto& particle: particles) {
		auto trObs = transformObservation(particle, observations);
		const std::vector<LandmarkObs> preds = predictLandmarks(particle, map_landmarks, sensor_range);
		dataAssociation(preds, trObs);
		particle.weight = 1.;
		for(const auto& obs: trObs) {
			const auto& pred = preds[obs.id];
			particle.weight *= multGaussProb(obs.x, obs.y, pred.x, pred.y, std_landmark);
			particle.associations.push_back(pred.id);
		}
		sumWeights += particle.weight;
		weights.push_back(particle.weight);
	}
	auto normalize = [=](Particle& particle)
	{
		particle.weight /= sumWeights;
	};
	std::for_each(particles.begin(), particles.end(), normalize);
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
	std::default_random_engine gen;
	std::vector<Particle> newParticles;
	for(size_t i = 0; num_particles > i; ++i) {
		newParticles.push_back(particles[distribution(gen)]);
	}
	particles = newParticles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations = associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

std::string ParticleFilter::getAssociations(Particle best)
{
	std::vector<int> v = best.associations;
	std::stringstream ss;
    std::copy( v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
std::string ParticleFilter::getSenseX(Particle best)
{
	std::vector<double> v = best.sense_x;
	std::stringstream ss;
    std::copy( v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
std::string ParticleFilter::getSenseY(Particle best)
{
	std::vector<double> v = best.sense_y;
	std::stringstream ss;
    std::copy( v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
