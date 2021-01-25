def category_parse(number):
    import deduplicator as dd
    number = int(number)
    #classes = ["person", "car", "truck", "bus", "motor", "bike", "rider", "traffic light", "traffic sign", "train"]
    return {0: dd.Categories.C_person,
            1: dd.Categories.C_car,
            3: dd.Categories.C_bus,
            4: dd.Categories.C_motorbike,
            5: dd.Categories.C_bycicle}.get(number, None)


def enu2GPS(x, y):
    import pymap3d as pm
    lat, lon, _ = pm.enu2geodetic(x, y, 0, 44.655540, 10.934315, 0)
    return lat, lon

def pixel2GPS(tif_file, x, y):
    from osgeo import gdal
    ds = gdal.Open(tif_file)
    adfGeoTransform = ds.GetGeoTransform()
    xoff = adfGeoTransform[0]
    a = adfGeoTransform[1]
    b = adfGeoTransform[2]
    yoff = adfGeoTransform[3]
    d = adfGeoTransform[4]
    e = adfGeoTransform[5]
    return d * x + e * y + yoff, a * x + b * y + xoff

def GPS2pixel(tif_file, lat, lon):
    from osgeo import gdal
    ds = gdal.Open(tif_file)
    adfGeoTransform = ds.GetGeoTransform()
    x = int(round((lon - adfGeoTransform[0]) / adfGeoTransform[1]))
    y = int(round((lat - adfGeoTransform[3]) / adfGeoTransform[5]))
    return x, y
