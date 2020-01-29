#include "ray.h"
#include "hittablelist.h"
#include "sphere.h"

#include <iostream>
#include <fstream>
#include <limits>

vec3 color(const ray& r, hittable *world) {
    hit_record rec;
    if (world->hit(r, 0.0, std::numeric_limits<float>::max(), rec)) {  // is a hit
        // calculate surface normal at hit location
        return 0.5*vec3(rec.normal.x()+1, rec.normal.y()+1, rec.normal.z()+1);
    }
    else {  // no hit, so render background
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5*(unit_direction.y() + 1.0);
        return (1.0-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
    }
}

int main() {
    // determine size of image
    int nx = 2000; // width
    int ny = 1000; // height

    // open file and add header
    std::ofstream outfile;
    outfile.open ("render.ppm");
    outfile << "P3\n" << nx << " " << ny << "\n255\n";

    vec3 lower_left_corner(-2.0, -1.0, -1.0);
    vec3 horizontal(4.0, 0.0, 0.0);
    vec3 vertical(0.0, 2.0, 0.0);
    vec3 origin(0.0, 0.0, 0.0);

    // generate world and place objects in it
    hittable *list[2];
    list[0] = new sphere(vec3(0,0,-1), 0.5);
    list[1] = new sphere(vec3(0,-100.5,-1), 100);
    hittable *world = new hittablelist(list,2);

    // render image
    for (int j = ny-1; j >= 0; j--) {   // for each pixel column in image
        for (int i = 0; i < nx; i++) {  // for each pixel in column
            float u = float(i) / float(nx);
            float v = float(j) / float(ny);
            ray r(origin, lower_left_corner + u*horizontal + v*vertical);

            vec3 p = r.point_at_parameter(2.0);
            vec3 col = color(r, world);

            int ir = int(255.99*col[0]);
            int ig = int(255.99*col[1]);
            int ib = int(255.99*col[2]);

            outfile << ir << " " << ig << " " << ib << "\n";
        }
    }
}