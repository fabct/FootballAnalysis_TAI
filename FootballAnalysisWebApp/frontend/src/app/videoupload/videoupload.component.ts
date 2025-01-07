import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';

@Component({
  selector: 'app-videoupload',
  templateUrl: './videoupload.component.html',
  styleUrls: ['./videoupload.component.css']
})
export class VideouploadComponent {

  selectedVideo: File | null = null;
  uploading: boolean = false; // Ajoutez cette variable pour gérer l'état de chargement

  constructor(private http: HttpClient, private router: Router) {}

  onVideoSelected(event: any) {
    this.selectedVideo = event.target.files[0];
  }

  uploadVideo() {
    if (this.selectedVideo) {
      const formData = new FormData();
      formData.append('video', this.selectedVideo);
      // Afficher l'icône de chargement
      this.uploading = true;

      this.http.post<any>('http://127.0.0.1:5000/upload', formData).subscribe(
        (response) => {
          console.log('Video uploaded successfully', response);
          // Traitez la réponse de votre backend si nécessaire
          // Masquer l'icône de chargement après la réussite
          this.router.navigate(['/videodownload']);
          this.uploading = false;
        },
        (error) => {
          console.error('Error uploading video', error);
          // Masquer l'icône de chargement après la réussite
          this.uploading = false;
        }
      );
    }
  }

}

