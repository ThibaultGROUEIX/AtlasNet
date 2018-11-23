#This script download the data from ENPC cloud

# The point clouds from ShapeNet, with normals
wget https://cloud.enpc.fr/s/j2ECcKleA1IKNzk/download --no-check-certificate
unzip download
rm download

# The corresponding normalized mesh (for the metro distance)
wget https://cloud.enpc.fr/s/RATKsfLQUSu0JWW/download --no-check-certificate
unzip download
rm download

# the rendered views
wget https://cloud.enpc.fr/s/S6TCx1QJzviNHq0/download --no-check-certificate
unzip download
rm download