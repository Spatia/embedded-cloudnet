use ndarray::{s, Array4};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    inputs,
};
use image::Luma;
use ort::value::TensorRef;

fn load_normalized_tiff(path: &str) -> Result<ndarray::Array2<f32>, Box<dyn std::error::Error>> {
    let img = image::open(path)?.into_luma16();
    let (width, height) = img.dimensions();
    
    let mut arr = ndarray::Array2::<f32>::zeros((height as usize, width as usize));
    for y in 0..height {
        for x in 0..width {
            arr[[y as usize, x as usize]] = img.get_pixel(x, y)[0] as f32 / 65535.0;
        }
    }
    Ok(arr)
}

fn threshold(threshold: f32, predictions_view: ndarray::ArrayView4<f32>, out_img: &mut image::GrayImage, height: usize, width: usize) {
    for y in 0..height {
        for x in 0..width {
            let logit = predictions_view[[0, 0, y, x]];
            let pixel_val = if logit > threshold { 255 } else { 0 };
            out_img.put_pixel(x as u32, y as u32, Luma([pixel_val]));
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file("../models/unet_dw_96k.onnx")?;

    let red_path = "../dataset/38-Cloud_test/test_red/red_patch_92_5_by_8_LC08_L1TP_029041_20160720_20170222_01_T1.TIF";
    let green_path = "../dataset/38-Cloud_test/test_green/green_patch_92_5_by_8_LC08_L1TP_029041_20160720_20170222_01_T1.TIF";
    let blue_path = "../dataset/38-Cloud_test/test_blue/blue_patch_92_5_by_8_LC08_L1TP_029041_20160720_20170222_01_T1.TIF";
    let nir_path = "../dataset/38-Cloud_test/test_nir/nir_patch_92_5_by_8_LC08_L1TP_029041_20160720_20170222_01_T1.TIF";

    let red = load_normalized_tiff(red_path)?;
    let green = load_normalized_tiff(green_path)?;
    let blue = load_normalized_tiff(blue_path)?;
    let nir = load_normalized_tiff(nir_path)?;

    let height = red.shape()[0];
    let width = red.shape()[1];

    let mut image = Array4::<f32>::zeros((1, 4, height, width));
    image.slice_mut(s![0, 0, .., ..]).assign(&red);
    image.slice_mut(s![0, 1, .., ..]).assign(&green);
    image.slice_mut(s![0, 2, .., ..]).assign(&blue);
    image.slice_mut(s![0, 3, .., ..]).assign(&nir);

    let input = TensorRef::from_array_view(image.view())?;
    let outputs = model.run(inputs![input])?;

    let output_tensor = outputs[0].try_extract_array::<f32>()?;
    let predictions_view = output_tensor.view().into_dimensionality::<ndarray::Ix4>()?;

    let mut out_img: image::GrayImage = image::ImageBuffer::new(width as u32, height as u32);

    threshold(2.0, predictions_view, &mut out_img, height, width);
     
    out_img.save("resultat_inference_rust.png")?;
    println!("Inference saved as 'resultat_inference_rust.png'.");

    Ok(())
}
