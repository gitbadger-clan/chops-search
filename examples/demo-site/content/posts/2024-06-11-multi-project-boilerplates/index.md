+++
title = "Using copier to aggregate multiple project boilerplates"
date = 2024-06-11
draft = false
+++

## Why?

Technically when you start a new project the best way to approach it is by using the `CLI` tool of the realm, such as `svelte-kit`, `astro`, `django-cli` etc..., you get the idea. The huge bonus to doing this is that you get the best practices baked in and as new standards are created the `CLI` gets updated.

So far the frontend has been a lot luckier with the tools as far as project generation goes, every major framework having come out with their own project generation tool, some having more than one possibly due to multiple schools of thought.

There are some backend frameworks that have project generation tools too but so far it seems to be difficult to agree on the structure. The best you can do is find a way to structure it that looks like the majority and makes sense for you. I have been building spiders and crawlers for data ingestion pipelines using `python` at first and then `node` and `go`. Even more recently I have been looking at hacking out some tweaks in some of my `neovim` plugins(that is lua).

For example neovim plugins have a pretty standard setup, they will have a folder layout something like the following:

```bash
scratcher.nvim/
├── README.md
├── lua
│   └── scratcher
│       └── init.lua
└── plugin
    └── scratcher.lua
```

The standard way of naming things seems to be gravitating towards having some conventions as you can see so setting up a new plugin would be pretty much repetitive and automatable. And it will probably save you some time and willpower in the long run, provided you have some sense of what your final architecture needs to look like.

![Copier clones](copier.png)

## How?

If you are coming from `python` like I am then you may already be familiar with [`cookiecutter`](https://github.com/cookiecutter/cookiecutter). I have been in the situation a few times where it might have made sense to use it, but every time it was a matter of balancing out the timeline and trying to stay away from over engineering.

Lately though the stuff I have been dealing with has been slightly on the more experimental side so churning out something new is something that happens quite often, so it makes more sense to have a prebaked architecture for specific project styles.

From the project templating libraries I was aware of [`cookiecutter`](https://github.com/cookiecutter/cookiecutter) and [`copier`](https://copier.readthedocs.io/en/stable/). `cookiecutter` uses `json` for driving the generation while `copier` uses `yaml`.

In the end I used copier as I tend to favor `yaml` because it allows for comments in the config. It makes it easy to plop random pieces of info or even docs in there. Since the config files driving the wizard it can become quite convoluted and also difficult to read as you would normal code so having the ability to document different options is probably a plus I would think.

The library allows you to build a sort of setup wizard where you can set up your desired flow of questions, and you can use the choices supplied to drive what folders will be used in the final project boilerplate. This is pretty nifty as it gives you the ability to customize stuff all the way down to the build process.

Another neat thing is that when you have decent chunks of code that can be shared, so you can just put that in your boilerplate, so it will essentially give you things just the way you like them.

## Cherry pick of features

There are a few notable features that I would kick myself if I didn't mention:

- you can define choices for your options by using the `choices` key in your `copier.yml` file:

  ```yaml
  project_type:
  type: str
  help: What type of project are you creating?
  default: neovim-plugin
  choices:
    - neovim-plugin
    - golang-cli
    - python-cli
  ```

- you can include different files in the root level `copier.yml` thus breaking down the wizard in composable parts, for example the CI/CD parts can be shared across projects

  ```yaml
  !include shared-conf/ci-cd.*.yml
  ```

- defaults are very powerful and can also make use of current runtime context which is quite nice. Essentially you can think of it as a way to have your very own project wizard that is tweaked for every one of your needs.

## Cool use-cases

- you can define a folder/file be created conditionally depending on an option selection eg:

  You define your `copier.yml` like this to give you a choice into the type of project:

  ```yaml
  project_type:
  type: str
  help: What type of project are you creating?
  default: neovim-plugin
  choices:
    - neovim-plugin
    - golang-cli
    - python-cli
  ```

  you then create a conditionally rendered folder, the naming follows `jinja` templating rules, so it might look something like the following

  ```bash
  {% if project_type == 'neovim-plugin' %}{{project_name}}.nvim{% endif %}
  ```

  It looks a bit strange, it will probably not work on Windows and it might look daunting at first but hopefully you will only need to revisit the hierarchy when you update your project structure template. This is not something I would expect to happen very often.

- in more advanced use cases you may need to write some custom python code for transforming/processing of entities in your `jinja` templates, or template strings.
  To hook this in you need to enable the `jinja` template extensions and add a separate package `copier-templates-extensions`

  ```yaml
  _jinja_extensions:
    - copier_templates_extensions.TemplateExtensionLoader
    - extensions/context.py:ContextUpdater
  ```

  This allows you to load specific extensions in your project generator runtime and it can serve different functions. The following snippet illustrates a way you can update the context:

  ```python
  from copier_templates_extensions import ContextHook


  class ContextUpdater(ContextHook):
      def hook(self, context):
          new_context = {}
          new_context["onboarding"] = "first steps with " + context["project-type"]
          return new_context
  ```

  For example you might decide to create a file called "first steps with <project-type>.txt" once the project is generated. In template form the file name would be `{{ onboarding }}`

## Conclusions

- you can build hybrid boilerplates for your project and drive generating the folder hierarchy from a single repo
- the notation is a bit weird with templated folder names, will not work on Win
- the templated naming is also very powerful allowing for conditional creation of folders
