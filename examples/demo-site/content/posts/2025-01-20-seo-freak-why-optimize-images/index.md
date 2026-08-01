+++
title = "How I optimized my blog images using Rust"
date = 2025-01-20
draft = false
[taxonomies]
tags = ["rust", "images", "seo", "performance"]
+++

## Why?

Ages ago I read a blog post about how images can be optimized and that gives visitors to your site a better experience as the site loads faster yadda, yadda. Why would anyone not want this?

My blog is built with [zola](https://www.getzola.org/) so I am a bit of a `rust` fanboy and want to use `rust` whenever it makes sense and my novice skills can handle it. Not going to lie, I am not a `rust` expert and for me `ChatGPT` is a useful tool as it most often than not points me in the right direction.

But I digress, a few days ago I decided to resurrect my blog, I tried to write a bit more consistently last year and got a streak of a few articles going, was feeling pretty good about it, but then my daughter was born and I was thrown in the gauntlet of figuring things out as a first time dad and the blog was left to rot. Since I am a bit more web marketing savvy, I decided to add some SEO to my blog, maybe I get some more visitors to it and get a sense of how popular it is.

I am trying to be polite with the people that land on my blog and not track them so I don't use cookies. I host my stuff on `cloudflare` since that gives the best bang for my buck. In other words I want my blog to be performant and free to host.

My blog uses some analytics that are available through [`cloudflare`](https://www.cloudflare.com/web-analytics/) but they are very respectful of user privacy in that they are `GDPR` and `CCPA` compliant. This saves me the hassle of having to add a cookie consent form that disrupts the user navigation experience. I both like and dislike the analytics from `cloudflare` as the numbers I am seeing are a bit weird as I am only seeing a constant number.

Since I learned a bit more about `SEO` and about [`Google Search Console`](https://search.google.com/search-console/about) I decided to check my blog's performance and see what I can do to improve it. Submitted my sitemap and ran a performance check and even if performance was at 100/100 I saw that the images were not optimized.

My OCD kicked in and I had to figure out a way to address it, especially since I remembered that I read [an article](https://endler.dev/2020/perf) talking about this. I dug into it a bit and noticed he is using `ImageMagik`, `cavif` and `cwebp` to optimize the images, I decided to go a different way, essentially almost reinventing the wheel. I built a `rust cli` that converts bigger `png` and `jpeg` images to `webp` or `cavif`.

![Optimization when you have no idea what to optimize](option1.png)

## How?

The step by step process looks like this:

- Change the shortcode for images to try and render the optimal image if supported by the browser

- write the rust tool that traverses the directory tree and convert images to `webp` or `avif`

- integrate the tool into the github action pipeline

- perform caching on the github workflow to avoid spending too many github minutes on the actual conversion

## What's in it for me?

### **1. Storage Cost Savings**

- **AVIF can be ~70% smaller than PNG** while maintaining similar or better quality.
- If you're storing images on **AWS S3, Google Cloud Storage, DigitalOcean Spaces, or another cloud provider**, reducing storage by **70%** directly cuts storage costs by the same percentage.
  **Example:**
  - **100GB of PNGs → ~30GB of AVIF**
  - If storage costs **$0.023 per GB (AWS S3 Standard)**:
    - PNG: **$2.30/month**
    - AVIF: **$0.69/month**
    - **Savings: ~$1.61 per 100GB/month (~70%)**

### **2. Egress Bandwidth Cost Savings**

- Most cloud providers charge for outbound bandwidth (data transferred to users).
- **Smaller AVIF files mean lower bandwidth usage, leading to significant savings.**
- **AVIF reduces bandwidth usage by ~70% compared to PNG.**
  **Example with AWS CloudFront:**
  - **Data transfer cost (to the internet):** **$0.085 per GB**
  - **If you serve 1TB of PNGs per month:**
    - **PNG: 1TB → $85/month**
    - **AVIF (70% smaller): 0.3TB → $25.50/month**
    - **Savings: ~$59.50 per TB/month (~70%)**

### **3. CDN Caching & Requests**

- Many e-commerce sites use a **CDN (Cloudflare, CloudFront, Fastly, etc.)**.
- Smaller images:
  - Improve **cache hit ratio** (more images fit in CDN cache).
  - Reduce **origin fetch requests**, further lowering egress costs.
  - Speed up load times, improving user experience.

### **Total Cost Savings Estimate**

| **Cost Factor**          | **PNG**          | **AVIF** | **Savings**      |
| ------------------------ | ---------------- | -------- | ---------------- |
| **Storage (100GB)**      | $2.30            | $0.69    | **$1.61 (70%)**  |
| **Egress (1TB/month)**   | $85.00           | $25.50   | **$59.50 (70%)** |
| **Total Savings per TB** | **$87.11/month** |          |                  |

## The Playbook

#### Step 1: Shortcode

I avoided using javascript for this since `html` already gives a mechanism to render an image with a fallback

```html
<picture>
  <source srcset="{{id}}.avif" {% if alt %}alt="{{alt}}" {% endif %} />
  <source srcset="{{id}}.webp" {% if alt %}alt="{{alt}}" {% endif %} />
  <img src="{{id}}.png" {% if alt %}alt="{{alt}}" {% endif %} />
</picture>
```

#### Step 2: Rust tool

The way I structure my posts is that each post lies neatly inside its own folder, along with all the images and any other extra assets that add some sort of value to the content.

So, from the theme I grab all the `png` images

```rust
    let mut input_paths: Vec<Params> = glob("content/**/*.png")?
        .filter_map(Result::ok)
        .map(|path| Params {
            path,
            should_recreate: args.recreate,
            ..Default::default()
        })
```

The trouble is the theme also contains some images which need to be converted. At the moment the only image is my logo

```rust
    let theme_image_paths: Vec<Params> = glob("themes/**/*.jpg")?
        .filter_map(Result::ok)
        .map(|path| Params {
            path,
            should_recreate: args.recreate,
            should_resize: true,
        })
        .collect();

```

In order to save time, converting images that exist already is a bit redundant so the tool checks if the path exists already and if it does, conversion is skipped. Working with paths is surprisingly straightforward. I was previously quite afraid to write code in `rust` because I feared the overhead.

The actual code is stupid easy to understand and reason about, even for me

```rust
    //  webp file path
    let webp_file_path = parent_dir.join(format!("{}.webp", file_stem).as_str()); // Convert to .webp as an example
    // was it already converted?
    if !webp_file_path.exists() {
```

A big chunk of the code lies in the conversion code which also gave me the most brain pain.
We converted `webp` using the [webp crate](https://docs.rs/webp/latest/webp/)

```rust
fn convert_to_webp(img: &DynamicImage, output_path: &str) -> AnyResult<()> {
    let encoder = WebpEncoder::from_image(img).unwrap();
    let webp_data = encoder.encode(75.0); // Quality 75
    let mut file = File::create(output_path)?;
    file.write_all(&webp_data)?;
    println!("Saved WebP to {}", output_path);
    Ok(())
}

```

And then `avif` using the [avif crate](https://crates.io/crates/ravif)

```rust
fn convert_to_avif(img: &DynamicImage, output_path: &str) -> AnyResult<()> {
    let (width, height) = img.dimensions();

    let rgba = img.to_rgba8();
    let encoded_avif = Encoder::new()
        .with_quality(50.0)
        .with_alpha_quality(50.0)
        .with_speed(10)
        .with_alpha_color_mode(AlphaColorMode::UnassociatedClean)
        .with_num_threads(Some(4));
    let avif_pixels = rgba
        .pixels()
        .map(|p| Rgba {
            r: p[0],
            g: p[1],
            b: p[2],
            a: p[3],
        })
        .collect::<Vec<Rgba>>();

    let EncodedImage {
        avif_file,
        color_byte_size,
        alpha_byte_size,
        ..
    } = encoded_avif
        .encode_rgba(Img::new(
            &avif_pixels,
            width.try_into().unwrap(),
            height.try_into().unwrap(),
        ))
        .unwrap();
    let mut file = File::create(output_path)?;
    file.write_all(&avif_file)?;
    println!("Saved AVIF to {}", output_path);
    Ok(())
}
```

The reason converting to `avif` was a bit more convoluted was due to the requirement for pixels to be in `rgba` format. I had to convert the image to `rgba` and then convert the pixels to `Rgba` format(thank you libs with different types). This was a bit of a pain but I managed to get it working.

I'm not an image processing expert so the solution was the result of a long conversation with trail and error with `ChatGPT`, then again this is why I love `rust` and how strict it is. It forces you to write code in a way that if it runs it most likely is correct.

#### Step 3: Github action

The action installs and enables `rust` so that the cli can be used

```yaml
- name: Install Rust
  run: |
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    rustup toolchain install nightly
    rustup default nightly
    echo "$HOME/.cargo/bin" >> $GITHUB_PATH

- name: Verify Rust Installation
  run: |
    rustup --version
    rustc --version
    cargo --version

- name: Build CLI tool
  run: |
    cargo build --manifest-path ./helpers/image-optimizer/Cargo.toml --release --verbose

- name: Perform the optimization
  run: ./helpers/image-optimizer/target/release/image-optimizer
```

One thing I particularly liked was the fact that it is possible to use a relative path to `Cargo.toml` which means no mucking about with paths.

#### Step 4: Caching

Now one thing about `rust` that is a bit of a bummer is that builds take quite some time. I guess that is the price to pay for static memory analysis. I would love to do a deep dive at some point on the optimization of rust build times but that is a story for another time.

The one thing that `github` tends to hold you accountable for is the number of build minutes you use when a workflow runs, so having rust install itself, download dependencies and then run a build for the tools can quickly add up.

The optimization for build times covers caching cargo dependencies but also caching the built binary.

I cached most things that I was able to but I am getting mixed results when trying to cache apt packages. It simply does not seem to work as intended in the naive approach.

```yaml
- name: Install OS Dependencies (if needed)
  run: |
    # Create a file listing your required packages, one per line.
    cat > apt-packages.txt << EOF
    nasm
    EOF
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends $(cat apt-packages.txt)

- name: Cache Rust toolchain
  uses: actions/cache@v3
  with:
    path: ~/.rustup
    key: ${{ runner.os }}-rustup-${{ hashFiles('rust-toolchain') }}
    restore-keys: |
      ${{ runner.os }}-rustup-

  ...
- name: Cache Cargo dependencies and target
  uses: actions/cache@v3
  with:
    path: |
      ~/.cargo/registry
      ~/.cargo/git
      ./helpers/image-optimizer/target
    key: ${{ runner.os }}-manual-cargo-${{ hashFiles('**/Cargo.lock') }}
```

The savings in time by using the caching is quite substantial. The first run of the workflow took 15 minutes, the run that had cached the deps was less than 1 minute. Even with a substantial amount of images this will most likely not be a bottleneck.

## Conclusion

- Optimizing images can decrease load on the server up to 70%
- ... and it also improves performance which is beneficial for SEO
- Optimizing github workflows can save you a lot of wait time
- ... and github minutes
- While this was interesting to do, I optimized for something that did not move the needle at all, it just made the evaluation in the `Google Search Console` a bit better.

This is what I have to blame my OCD for. I am happy with the result and I learned quite a few things about `rust` and `image` processing. I am also happy that I managed to get the `avif` conversion working as it is a format that is not yet widely supported but is the most efficient format out there.
