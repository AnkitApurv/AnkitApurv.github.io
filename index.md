---
layout: default
permalink: "/"
title: Home
---

<dl>
  {% for post in site.posts %}
    <dt>{{ post.categories }} - <a href="{{ post.url }}">{{ post.title }}</a></dt>
    <dd>{{ post.excerpt }}</dd>
    <br/>
  {% endfor %}
</dl>