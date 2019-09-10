<template>
  <div class="hello md-alignment-center">
    <h1>{{ title }}</h1>
    <h2>{{ authors }}</h2>

    <div class="md-layout md-gutter">
      <div class="md-layout-item linkItems" v-for="value in linkList" :key="value.id">
        <a v-bind:href="value.link">{{value.link_name}}</a>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: "HeaderComponent",
  props: {
    title: String,
    authors: String,
    linkList: Object
  }
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
h1 {
  margin: 40px 40px 40px, 40px;
  text-align: center;
}
h2 {
  margin: 40px 40px 40px, 40px;
  text-align: center;
}
.linkItems {
  text-align: center;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}
a {
  color: #42b983;
}
</style>
