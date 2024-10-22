#ifndef SERVER_H
#define SERVER_H

#include <chrono>
#include <cstdio>
#include <httplib.h>
#include <string>
#define SERVER_CERT_FILE "./cert.pem"
#define SERVER_PRIVATE_KEY_FILE "./key.pem"


std::string dump_headers(const httplib::Headers &headers);

std::string log(const httplib::Request &req, const httplib::Response &res);

void handleSingleCompute(const httplib::Request &req, httplib::Response &res);
void handleComputePer6h(const httplib::Request &req, httplib::Response &res);
void handleComputeFeatures(const httplib::Request &req, httplib::Response &res);
void handleAtomFeatures(const httplib::Request &req, httplib::Response &res);

#endif