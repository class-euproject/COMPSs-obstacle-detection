def category_parse(number):
    import deduplicator as dd
    number = int(number)
    #classes = ["person", "car", "truck", "bus", "motor", "bike", "rider", "traffic light", "traffic sign", "train"]
    return {0: dd.Categories.C_person,
            1: dd.Categories.C_car,
            3: dd.Categories.C_bus,
            4: dd.Categories.C_motorbike,
            5: dd.Categories.C_bycicle}.get(number, None)


def pixel2GPS(x, y):
    import pymap3d as pm
    lat, lon, _ = pm.enu2geodetic(x, y, 0, 44.655540, 10.934315, 0)
    return lat, lon