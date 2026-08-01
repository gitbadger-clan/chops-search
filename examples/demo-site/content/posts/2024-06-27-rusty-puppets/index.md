+++
title = "Scraping with rust and headless chrome (Part I)"
date = 2024-06-27
draft = false
[taxonomies]
tags = ["rust", "web scraping", "headless chrome"]
+++

## Why?

Rust is pretty amazing but there are a few things that you might be weary about. There are [few war stories](https://discord.com/blog/why-discord-is-switching-from-go-to-rust) of companies building their entire stack on `rust` or and then living happily ever after. Software is an ever evolving organism so in the [darwinian sense the more adaptable the better](https://www.darwinproject.ac.uk/people/about-darwin/six-things-darwin-never-said/evolution-misquotation). Enough of that though, not here to advocate any particular language or framework, what I want is to share my experience with writing an equivalent scraper in `rust` to [my previous post](../using-go-chromedp/) where I used `golang` and `chromedp`.

The experience using `go` with `chromedp` to automate chrome was pretty good, it is not as powerful as what is available in `puppeteer` so I figured I would have a look at what might be available in the `rust` landscape.

## What?

![Puppeteer puppeteering rusty humans](rusty-puppets.png)

In `rust` there are several libraries that deal with browser automation, a few I have had a look at are:

- [fantocini](https://github.com/jonhoo/fantoccini) - A high-level API for programmatically interacting with web pages through WebDriver, but I want chrome devtools protocol instead.
- [rust-headless-chrome](https://github.com/rust-headless-chrome/rust-headless-chrome) - chrome devtools protocol client library in rust, not as active as the crate I wound up using.
- [chromiumoxide](https://github.com/mattsse/chromiumoxide) - this is the one that seem to be the most active in terms of development so it looks like a good choice at time of writing.

## TL;DR

As I was reading one of my older posts that focuses on quasi live coding I realized it was boring as hell, and if your attention span is that of a goldfish, like mine is, it would probably make sense to just drop in a link to the [repo](https://github.com/adaschevici/rustic-toy-chest/tree/main/rust-crawl-pupp) so that you can download the code and try it out yourself. The repo is a collection of rust prototypes that I have been building for fun and learning, haven't had yet a compelling reason to use rust in production unfortunately :cry:.

## How?

To my surprise the code was closer in structure to the [`puppeteer`](https://pptr.dev/) version than it was to the [`chromedp`](https://github.com/chromedp/chromedp). The `chromedp` version uses nested context declarations to manage the browser and page runtimes, the `rust` version uses a more linear approach. You construct a browser instance and then you can interact with it as a user would. This points at the fact that the `chromiumoxide` api is higher level.

The way you can set things up to keep your use cases separate is by adding [`clap`](https://docs.rs/clap/latest/clap/) to your project and use command line flags to select the use case you want to run.

You will see that I have covered most cases but not everything is transferable from `puppeteer` or `chromedp` to the `chromiumoxide` version. I will not go through the setup of `rustup`, rust toolchain or `cargo` as this is a basic and well documented process, all you have to do is search for `getting started with rust` and you will find a bunch of resources.

## Show me the code

#### 1. Laying down the foundation

- set up my project root

  ```bash
  cargo new rust-crawl-pupp
  cd rust-crawl-pupp
  cargo install cargo-edit # this is useful for adding and upgrading dependencies
  ```

- add dependencies via `cargo add`

  ```toml
  [dependencies]
  chromiumoxide = { version = "0.5.7", features = ["tokio", "tokio-runtime"] } # this is the main dependency
  chromiumoxide_cdp = "0.5.2" # this is the devtools protocol
  clap = { version = "4.5.7", features = ["derive", "cargo"] } # this is for command line parsing
  futures = "0.3.30" # this is for async programming
  tokio = { version = "1.38.0", features = ["full"] } # this is the async runtime
  tracing = "0.1.40" # this is for logging
  tracing-subscriber = { version = "0.3.18", features = ["registry", "env-filter"] } # this is for logging
  ```

- add `clap` command line parsing to the project so that each different use case can be called via a subcommand
  define your imports

  ```rust
  use clap::{Parser, Subcommand};
  ```

  define your command structs for parsing the command line arguments, this will allow for each use case to be called with its own subcommand like so `cargo run -- first-project`, `cargo run -- second-project`, and so on.

  ```rust
  #[derive(Parser)]
  #[command(
      name = "OxideCrawler",
      version = "0.1",
      author = "artur",
      about = "An example application using clap"
  )]
  struct Cli {
      #[command(subcommand)]
      command: Commands,
  }
  #[derive(Subcommand, Debug)]
  enum Commands {
      FirstProject {},
      SecondProject {},
      ...
  }
  ```

  the way you can hook this into the main function is via a `match` statement that will call the appropriate function based on the subcommand that was passed in.

  ```rust
  let args = Cli::parse();
  ...
  match &args.command {
      Commands::FirstProject {} => {
          let user_agent = spoof_user_agent(&mut browser).await?;
          info!(user_agent, "User agent detected");
      }
      ...
  }

  ```

#### 2. Starting browser and the browser cleanup

- use the `launch` method and its options to start the browser, if the viewport and window size are different, the browser will start in windowed mode, with the page size being smaller.

  ```rust
  let (mut browser, mut handler) = Browser::launch(
      BrowserConfig::builder()
          .with_head() // this will start the browser in headless mode
          .no_sandbox() // this will disable the sandbox
          .viewport(None) // this will set the viewport size
          .window_size(1400, 1600) // this will set the window size
          .build()?,
  )
  .await?;

  let handle = tokio::task::spawn(async move {
      loop {
          match handler.next().await {
              Some(h) => match h {
                  Ok(_) => continue,
                  Err(_) => break,
              },
              None => break,
          }
      }
  });
  ```

- the browser cleanup needs to be done correctly and there are two symptoms that you will see if you missed anything:

  - the browser will not close - hangs at the end
  - you might get a warning like the following:

    ```bash
    2024-06-26T08:40:01.418414Z  WARN chromiumoxide::browser: Browser was not closed manually, it will be killed automatically in the background
    ```

    to correctly clean up your browser instance you will have to call these on the code paths that close the browser

  ```rust
  browser.close().await?;
  browser.wait().await?;
  handle.await?;
  ```

#### 3. Use cases

In the [repo](https://github.com/adaschevici/rustic-toy-chest/tree/main/rust-crawl-pupp) each use case lives in its own module most of the time. There are some cases where you might have two living in the same module when they are very closely related, like in Use Case `c.`.

**a. Spoof your user agent:**

The only way I have found to set your user agent was from the [`Page`](https://docs.rs/chromiumoxide/latest/chromiumoxide/page/struct.Page.html#) module via the `set_user_agent` method

```rust
let page = browser.new_page("about:blank").await?;
page.set_user_agent(
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) \
     Chrome/58.0.3029.110 Safari/537.36",
)
.await?;
page.goto("https://www.whatismybrowser.com/detect/what-is-my-user-agent")
    .await?;

```

**b. Grabbing the full content of the page** is pretty straightforward

```rust
    let page = browser
      .new_page("https://scrapingclub.com/exercise/list_infinite_scroll/")
      .await?;
  let content = page.content().await?;
```

**c. Grabbing elements via css selectors**,

```rust
let elements_on_page = page.find_elements(".post").await?;
let elements = stream::iter(elements_on_page)
    .then(|e| async move {
        let el_text = e.inner_text().await.ok();
        match el_text {
            Some(text) => text,
            None => None,
        }
    })
    .filter_map(|x| async { x })
    .collect::<Vec<_>>()
    .await;
```

performing **relative selection** from a specific node and mapping the content to `rust` types

```rust
...
let product_name = e.find_element("h4").await?.inner_text().await?.unwrap();
let product_price = e.find_element("h5").await?.inner_text().await?.unwrap();
Ok(Product {
    name: product_name,
    price: product_price,
})
...

```

**d. When the page has infinite scroll** you will have to scroll to the bottom of the page to be able to collect all the elements you are interested in. To achieve this you need to inject `javascript` into the page context and trigger a run of the function. The `chromiumoxide` api seems to have really decent support for this, I faced much less resistance than I did with `chromedp` and `go`.

```rust
let js_script = r#"
    async () => {
      await new Promise((resolve, reject) => {
        var totalHeight = 0;
        var distance = 300; // should be less than or equal to window.innerHeight
        var timer = setInterval(() => {
          var scrollHeight = document.body.scrollHeight;
          window.scrollBy(0, distance);
          totalHeight += distance;

          if (totalHeight >= scrollHeight) {
            clearInterval(timer);
            resolve();
          }
        }, 500);
      });
  }
"#;
let page = browser
    .new_page("https://scrapingclub.com/exercise/list_infinite_scroll/")
    .await?;
page.evaluate_function(js_script).await?;
```

**e. When you need to wait for an element to load**, this was not exactly part of the `chromiumoxide` api so I had to hack it together. Given my limited rust expertise there probably a better way to do this but this is what I managed to come up with. If the async block runs over the timeout then the `element_result` will be an error, otherwise poll the dom for the element we are looking for.

```rust
use tokio::time::{timeout, Duration};
...
let element_result = timeout(timeout_duration, async {
    loop {
        match page.find_element(selector).await {
            Ok(element) => return Ok(element),
            // Wait for a short interval before checking again
            Err(e) => tokio::time::sleep(Duration::from_millis(100)).await,
        }
    }
})
.await;
```

#### 4. Fixtures to replicate various scenarios

Some websites, actually most websites have some sort of delay for loading different parts of the page, in order to prevent blocking the entire page. To replicate this behavior fixtures can be used to inject nodes into the dom with a delay. For the more edge case scenarios I created fixtures to emulate edge behaviors while not actually having to remember a website that is live and behaves like that.

The HTML is really basic:

```html
<div id="container">
  <!-- New node will be appended here -->
</div>

<script src="script.js"></script>
```

The `script.js` file is slightly more, but still fairly straightforward:

```javascript
document.addEventListener("DOMContentLoaded", () => {
  // Function to create and append the new node
  function createDelayedNode() {
    // Create a new div element
    const newNode = document.createElement("div");

    // Add some content to the new node
    newNode.textContent = "This is a new node added after a delay.";

    // Add some styles to the new node
    newNode.style.padding = "10px";
    newNode.style.marginTop = "10px";
    newNode.style.backgroundColor = "#f0f0f0";
    newNode.style.border = "1px solid #ccc";
    newNode.id = "come-find-me";

    // Append the new node to the container
    const container = document.getElementById("container");
    container.appendChild(newNode);
  }

  // Set a delay (in milliseconds)
  const delay = 3000; // 3000ms = 3 seconds

  // Use setTimeout to create and append the node after the delay
  setTimeout(createDelayedNode, delay);
});
```

What it will do is create a new node with some text content and some styles, then append it to the container div after a delay of 3 seconds.

## Why to be continued?

What I hate more than `to be continued` in a TV show where I don't have the next episode available is a blog post that has code that looks reasonable and that it might work, but doesn't. So going by the lesser of two evils principle I decided to make this a two parter which will give me the time to write and test the other use cases in order to make sure everything works as expected.

## Conclusions

- This is one of the few times I have stuck with `rust` through the pain and I have to say it was a better experience than I had with `go` and `chromedp`
- writing the code was slightly faster since there was less boilerplate to write
- messing around with wrappers and `unwrap()` was challenging but probably in time it gets easier
- the code in `rust` looks more like `puppeteer` than the `go` version did

#### In Part II I will cover dealing with bot protection, handling frames, forms and more. Stay tuned
