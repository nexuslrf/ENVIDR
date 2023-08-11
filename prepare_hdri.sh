#!/bin/bash
url=https://github.com/google/filament/raw/main/third_party/environments/
dir=data/env_sphere/hdri
mkdir -p $dir

env_list=(
    "flower_road_2k"
    "flower_road_no_sun_2k"
    "graffiti_shelter_2k"
    "lightroom_14b"
    "noon_grass_2k"
    "parking_garage_2k"
    "pillars_2k"
    "studio_small_02_2k"
    "syferfontein_18d_clear_2k"
    "the_sky_is_on_fire_2k"
    "venetian_crossroads_2k"
)

for env in "${env_list[@]}";
do
    echo "Downloading $env"
    wget -nc -P $dir $url/$env.hdr
done

# convert hdri to ktx
filament_dir=data/env_sphere/filament
mkdir -p $filament_dir
wget -nc -P $filament_dir https://github.com/google/filament/releases/download/v1.30.0/filament-v1.30.0-linux.tgz
tar -xzf $filament_dir/filament-v1.30.0-linux.tgz -C $filament_dir
filament_bin=$filament_dir/filament/bin
ktx_dir=data/env_sphere/env_ktx

for env in "${env_list[@]}";
do
    echo "Converting $env"
    mkdir -p $ktx_dir/$env
    $filament_bin/cmgen -x $ktx_dir/$env --quiet --format=ktx $dir/$env.hdr
done
