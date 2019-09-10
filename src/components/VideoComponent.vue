<template>
  <div class="md-layout md-gutter">
    <div class="md-layout-item videoDiv">
      <iframe
        class="md-image"
        v-bind:width="width"
        v-bind:height="height"
        v-bind:src="videoLink"
        frameborder="0"
        allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
        allowfullscreen
      ></iframe>
    </div>
  </div>
</template>
<script>
export default {
  name: "VideoComponent",
  props: {
    videoLink: String,
    width: {
      default: 600
    },
    height: {
      default: 300
    }
  }
};
</script>
<style scoped>
.videoDiv {
  display: flex;
  align-items: center;
  justify-content: center;
}
</style>