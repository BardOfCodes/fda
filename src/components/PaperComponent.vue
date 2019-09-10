<template>
  <div class="hello">
    <div class="md-layout">
      <div class="md-layout-item paperDiv">
        <h1>Abstract</h1>
        <p>{{ paperContent.abstract}}</p>
        <h1>BibTex</h1>
        <div class="well">
          <p>{{ paperContent.bibtex }}</p>
        </div>
        <h1>Selected Results</h1>
        <img width="100%" src="../assets/table.png" />
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: "PaperComponent",
  props: {
    paperContent: Object
  }
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
h3 {
  margin: 40px 0 0;
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
.paperDiv {
  align-items: center;
  justify-content: center;
}
</style>
