<template>
  <div class="page-container main-container-centered">
    <!-- The Header Component -->
    <md-app>
      <md-app-content>
        <div class="md-layout md-gutter">
          <div class="md-layout-item md-content">
            <HeaderComponent
              v-bind:title="data.title"
              v-bind:authors="data.authors"
              v-bind:linkList="data.linkList"
            />
          </div>
        </div>

        <!-- The Video Component -->
        <div class="md-layout md-gutter">
          <div class="md-layout-item md-content">
            <VideoComponent v-bind:videoLink="data.videoLink" />
          </div>
        </div>

        <!-- The Paper Component -->
        <div class="md-layout md-gutter">
          <div class="md-layout-item md-content">
            <PaperComponent v-bind:paperContent="data.paperContent" />
          </div>
        </div>
      </md-app-content>
    </md-app>
  </div>
</template>

<script>
import HeaderComponent from "./components/HeaderComponent.vue";
import VideoComponent from "./components/VideoComponent.vue";
import PaperComponent from "./components/PaperComponent.vue";
import data_json from "./json/data.json";

export default {
  name: "app",
  components: {
    HeaderComponent,
    VideoComponent,
    PaperComponent
  },
  data() {
    return {
      data: data_json
    };
  }
};
</script>

<style>
@import url("https://fonts.googleapis.com/css?family=Roboto:400,500,700,400italic|Material+Icons");

html,
body {
  font-family: "Roboto Mono", monospace !important;
  font-weight: 400;
}

#app {
  font-family: "Roboto Mono", monospace !important;
  font-weight: 400;
}
.md-content {
  padding: 16px;
}
.page-container {
  max-width: 1100px;
  margin: 0 auto;
}
</style>
