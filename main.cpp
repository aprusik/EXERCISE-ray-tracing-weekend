#include "ray.h"
#include "hittablelist.h"
#include "sphere.h"
#include "random.h"
#include "camera.h"

#include <iostream>
#include <fstream>
#include <limits>

vec3 random_in_unit_sphere() {
    vec3 p;
    do { // find random point in cube and continue until point within sphere
        p = 2.0*vec3(random_double(), random_double(), random_double()) - vec3(1,1,1);
    } while (p.squared_length() >= 1.0);
    return p;
}

vec3 color(const ray& r, hittable *world) {
    hit_record rec;
    if (world->hit(r, 0.001, std::numeric_limits<float>::max(), rec)) {  // is a hit
        vec3 target = rec.p + rec.normal + random_in_unit_sphere();
        return 0.5 * color(ray(rec.p, target - rec.p), world);
    }
    else {  // no hit, so render background
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5*(unit_direction.y() + 1.0);
        return (1.0-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
    }
}

int main() {
    // determine size of image
    int nx = 1000; // width
    int ny = 500; // height
    int ns = 100; // number of antialiasing samples

    // open file and add header
    std::ofstream outfile;
    outfile.open ("render.ppm");
    outfile << "P3\n" << nx << " " << ny << "\n255\n";

    // generate world and place objects in it
    hittable *list[2];
    list[0] = new sphere(vec3(0,0,-1), 0.5);
    list[1] = new sphere(vec3(0,-100.5,-1), 100);
    hittable *world = new hittablelist(list,2);

    // render image
    camera cam;
    for (int j = ny-1; j >= 0; j--) {   // for each pixel column in image
        for (int i = 0; i < nx; i++) {  // for each pixel in column
            vec3 col(0, 0, 0);
            for (int s = 0; s < ns; s++) {
                float u = float(i + random_double()) / float(nx);
                float v = float(j + random_double()) / float(ny);
                ray r = cam.get_ray(u, v);
                col += color(r, world);
            }
            col /= float(ns);

            // gamma correction
            col = vec3( sqrt(col[0]), sqrt(col[1]), sqrt(col[2]) );

            int ir = int(255.99*col[0]);
            int ig = int(255.99*col[1]);
            int ib = int(255.99*col[2]);

            outfile << ir << " " << ig << " " << ib << "\n";
        }
    }
}